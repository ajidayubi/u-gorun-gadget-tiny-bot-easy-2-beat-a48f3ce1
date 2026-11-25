"""Tiny two-sleeve trading bot for hourly Solana prices (bars-replay/v1).

Strategy, in plain words: two small sleeves share one account, each allowed
at most a quarter of the account per coin, and the rest of the money sits in
USDC cash whenever neither sleeve sees anything worth buying.

Momentum sleeve — ride what is already rising. Every hour, for each of the
7 coins, compare a fast average (8 bars) with a slow average (24 bars) and
look at how much the coin rose over the last 12 bars and the last 24 bars.
A coin qualifies when its fast average sits above its slow average, it rose
more than 1% over the last 12 bars, and it rose over the last 24 bars.
Hold the best two qualifiers, 25% of the account each. Coins already held
stay held while they qualify: a newcomer only displaces the weakest held
coin when its 12-bar rise beats the held coin's by more than 3%, and no
coin is sold for rotation within 12 bars of buying it, so hourly noise
cannot spin the portfolio. A held coin is sold when its close falls 8%
below the highest close seen since buying it.

Dip sleeve — buy sharp falls, sell the bounce. A coin qualifies when it
fell more than 3% over the last 12 bars, sits more than 2% below its
24-bar slow average, and has already bounced at least 0.5% off its low of
the last 6 bars, so the bot never catches a knife still falling. Hold the
three deepest such dips, 25% of the account each. Sell a dip when it
bounces 2% above the entry price, falls 2% below it, or after 12 bars
held, whichever comes first.

Both sleeves: only trade when a position is off its target by more than
0.5% of the account, so small wiggles do not burn fees. Buys are paid
from cash on hand only; a sell in the same hour never funds a buy. A coin
sold on a trailing stop is left alone for 6 bars. Long only, no leverage,
no clock, no randomness: the same bars always give the same orders, so
the two grading runs agree byte for byte.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bars  # noqa: E402

FAST_LEN = 8
SLOW_LEN = 24
MOM_LEN = 12
MOM2_LEN = 24
MIN_BARS_M = 13
TOP_M = 2
MIN_MOM_BPS = 100          # 12-bar rise must beat +1%
RANK_GAP_BPS = 300         # a newcomer must beat the weakest held coin by 3%
MIN_HOLD_BARS = 12         # no rotation sale within 12 bars of buying
TRAIL_BPS = 800            # sell when 8% below the peak since entry
DIP_BPS = -300             # 12-bar fall must beat -3%
STRETCH_BPS = -200         # close must sit 2% below the slow average
REC_BPS = 50               # and 0.5% above its low of the last 6 bars
TOP_D = 3
TP_BPS = 200               # sell a dip on a 2% bounce above entry
SL_BPS = -200              # or a 2% fall below entry
MAX_HOLD = 12              # or after 12 bars held
WEIGHT_BPS = 2500          # 25% of equity per coin (the harness cap)
COOLDOWN_BARS = 6
MIN_CHG_BPS = 50           # ignore weight drift under 0.5% of equity
MAX_SLIPPAGE_BPS = 200


def _ema(prev, price, length):
    alpha = 2.0 / (length + 1)
    return price if prev is None else prev + alpha * (price - prev)


class Bot:
    def __init__(self):
        self.closes = {}    # symbol -> list of float closes
        self.fast = {}      # symbol -> fast average or None
        self.slow = {}      # symbol -> slow average or None
        self.peak = {}      # symbol -> highest close since entry
        self.entry = {}     # symbol -> close first seen held
        self.entered = {}   # symbol -> seq first seen held
        self.stopped = {}   # symbol -> seq of the trailing-stop sale
        self.owner = {}     # symbol -> 'mom' or 'dip'

    def on_batch(self, state, batch):
        if state.warming_up:
            return []
        seq = state.seq
        orders = []
        # Learn this bar's closes, in a fixed symbol order.
        for symbol in sorted(state.symbols):
            close = state.close(symbol)
            if close is None:
                continue
            price = close / (10 ** state.symbols[symbol]["price_scale"])
            self.closes.setdefault(symbol, []).append(price)
            self.fast[symbol] = _ema(self.fast.get(symbol), price, FAST_LEN)
            self.slow[symbol] = _ema(self.slow.get(symbol), price, SLOW_LEN)
        # Track peaks, entries and entry bars for what is held.
        for symbol in list(self.peak):
            if symbol not in state.positions:
                del self.peak[symbol]
        for symbol in list(self.entry):
            if symbol not in state.positions:
                del self.entry[symbol]
        for symbol in list(self.entered):
            if symbol not in state.positions:
                del self.entered[symbol]
                self.owner.pop(symbol, None)
        for symbol in state.positions:
            close = state.close(symbol)
            if close is None:
                continue
            price = close / (10 ** state.symbols[symbol]["price_scale"])
            self.peak[symbol] = max(self.peak.get(symbol, price), price)
            self.entry.setdefault(symbol, price)
            self.entered.setdefault(symbol, seq)
        # Rank the coins worth riding upward.
        ranked = []
        for symbol in sorted(state.symbols):
            hist = self.closes.get(symbol, [])
            if len(hist) < MIN_BARS_M:
                continue
            if self.owner.get(symbol) == "dip":
                continue
            if seq - self.stopped.get(symbol, -10 ** 9) <= COOLDOWN_BARS:
                continue
            fast = self.fast.get(symbol)
            slow = self.slow.get(symbol)
            if fast is None or slow is None or fast <= slow:
                continue
            mom = (hist[-1] / hist[-1 - MOM_LEN] - 1.0) * 10000
            if mom <= MIN_MOM_BPS:
                continue
            mom2 = (hist[-1] / hist[-1 - MOM2_LEN] - 1.0) * 10000 \
                if len(hist) > MOM2_LEN else 0
            if mom2 <= 0:
                continue
            ranked.append((mom, symbol))
        ranked.sort(reverse=True)
        mom_map = dict((s, m) for m, s in ranked)
        held_m = [s for _, s in ranked if s in state.positions]
        new_m = [s for _, s in ranked if s not in state.positions]
        mom_targets = list(held_m[:TOP_M])
        for symbol in new_m:
            if len(mom_targets) >= TOP_M:
                break
            mom_targets.append(symbol)
        if len(mom_targets) >= TOP_M and new_m:
            worst = min(mom_targets, key=lambda s: mom_map[s])
            best = new_m[0]
            if best not in mom_targets and mom_map[best] - mom_map[worst] > RANK_GAP_BPS:
                mom_targets.remove(worst)
                mom_targets.append(best)
        # Rank the deepest dips worth buying.
        dips = []
        for symbol in sorted(state.symbols):
            hist = self.closes.get(symbol, [])
            if len(hist) < 25:
                continue
            if self.owner.get(symbol) == "mom":
                continue
            if seq - self.stopped.get(symbol, -10 ** 9) <= COOLDOWN_BARS:
                continue
            slow = self.slow.get(symbol)
            if slow is None:
                continue
            ret12 = (hist[-1] / hist[-13] - 1.0) * 10000
            stretch = (hist[-1] / slow - 1.0) * 10000
            if REC_BPS and hist[-1] < min(hist[-7:-1]) * (1.0 + REC_BPS / 10000.0):
                continue
            if ret12 < DIP_BPS and stretch < STRETCH_BPS:
                dips.append((ret12, symbol))
        dips.sort()
        dip_targets = [s for _, s in dips[:TOP_D]]
        mom_set = set(mom_targets)
        equity = state.equity
        min_chg = equity * MIN_CHG_BPS // 10000
        # Sell first: dip exits, momentum trailing stops and rotations.
        for symbol in sorted(state.positions):
            holding = state.positions[symbol]
            close = state.close(symbol)
            if close is None:
                continue
            price = close / (10 ** state.symbols[symbol]["price_scale"])
            own = self.owner.get(symbol, "mom")
            sell = False
            if own == "dip":
                entry = self.entry.get(symbol, price)
                ret = (price / entry - 1.0) * 10000
                held_bars = seq - self.entered.get(symbol, seq)
                if ret >= TP_BPS or ret <= SL_BPS or held_bars >= MAX_HOLD:
                    sell = True
            else:
                peak = self.peak.get(symbol, price)
                if price < peak * (1.0 - TRAIL_BPS / 10000.0):
                    self.stopped[symbol] = seq
                    sell = True
                elif symbol not in mom_set:
                    if seq - self.entered.get(symbol, seq) >= MIN_HOLD_BARS:
                        sell = True
            if sell:
                if own == "mom" and price < self.peak.get(symbol, price) * (1.0 - TRAIL_BPS / 10000.0):
                    self.stopped[symbol] = seq
                orders.append({"side": "sell", "symbol": symbol,
                               "qty": holding.qty,
                               "max_slippage_bps": MAX_SLIPPAGE_BPS})
        # Then buy the targets that are underweight, from cash on hand.
        cash = state.cash
        for symbol, own in [(s, "mom") for s in mom_targets] + \
                          [(s, "dip") for s in dip_targets]:
            target_value = equity * WEIGHT_BPS // 10000
            held = state.positions.get(symbol)
            held_value = state.value(symbol, held.qty) if held else 0
            need = target_value - held_value
            if need <= min_chg:
                continue
            room = state.weight_room(symbol)
            spend = min(need, room, cash)
            min_notional = int(state.hello["fees"]["min_notional_quote_units"])
            if spend < min_notional or spend <= min_chg:
                continue
            spend = int(spend)
            if spend <= 0:
                continue
            orders.append({"side": "buy", "symbol": symbol,
                           "notional_quote_units": spend,
                           "max_slippage_bps": MAX_SLIPPAGE_BPS})
            self.owner[symbol] = own
            cash -= spend
        return orders


def main():
    bot = Bot()
    makefun_bars.run(bot.on_batch)


if __name__ == "__main__":
    main()
