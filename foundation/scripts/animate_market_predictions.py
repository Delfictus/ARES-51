#!/usr/bin/env python3
"""
Animate market close vs 2-min-ahead predictions using the CSV produced by
examples/historical_minute_resonance_prediction.rs.

Output: artifacts/market_predictions.gif
"""
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

CSV_PATH = os.path.join('artifacts', 'market_predictions.csv')
OUT_GIF = os.path.join('artifacts', 'market_predictions.gif')

def parse_csv(path):
    history = []
    forecast = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
            phase = row['phase']
            close = row['close'] and float(row['close']) or None
            pred = float(row['predicted_2m'])
            if phase == 'history':
                history.append((ts, close, pred))
            else:
                forecast.append((ts, None, pred))
    history.sort(key=lambda x: x[0])
    forecast.sort(key=lambda x: x[0])
    return history, forecast

def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"CSV not found: {CSV_PATH}. Run the Rust example first.")

    hist, fc = parse_csv(CSV_PATH)
    times = [t for t, _, _ in hist]
    closes = [c for _, c, _ in hist]
    preds = [p for _, _, p in hist]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title('2-min Ahead Predictions')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')

    line_close, = ax.plot([], [], label='Close', color='blue', linewidth=1.0)
    line_pred, = ax.plot([], [], label='Predicted +2m', color='orange', linewidth=1.0)
    ax.legend(loc='upper left')

    # Extend x-range with forecast horizon
    x_all = times + [t for t, _, _ in fc]
    y_all = closes + [preds[-1] if preds else 0.0] * len(fc)
    ax.set_xlim(min(x_all), max(x_all))
    ymin = min([c for c in closes if c is not None] + preds)
    ymax = max([c for c in closes if c is not None] + preds)
    pad = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)

    def update(i):
        # Animate through history first, then forecast
        if i < len(times):
            line_close.set_data(times[:i+1], [c for c in closes[:i+1]])
            line_pred.set_data(times[:i+1], preds[:i+1])
        else:
            k = i - len(times)
            # hold last known close level; extend predictions with forecast points
            fc_times = [t for t, _, _ in fc[:k+1]]
            line_close.set_data(times, closes)
            line_pred.set_data(times + fc_times, preds + [p for _, _, p in fc[:k+1]])
        return line_close, line_pred

    total_frames = len(times) + len(fc)
    ani = FuncAnimation(fig, update, frames=total_frames, interval=20, blit=True)

    print(f"Writing GIF → {OUT_GIF}")
    writer = PillowWriter(fps=30)
    ani.save(OUT_GIF, writer=writer)
    print("Done.")

if __name__ == '__main__':
    main()

