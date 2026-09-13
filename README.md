OP.exe

Single-page Streamlit market-analysis dashboard. It combines the user's MNQ
15-minute-to-1-minute FVG retracement setup with multi-timeframe market structure,
BOS, liquidity sweeps, SMA20/50, momentum, volume, FVG/IFVG, catalyst headlines,
session/freshness gates, and capped risk planning.

Render

Build command: pip install -r requirements.txt

Start command: streamlit run main.py --server.port $PORT --server.address 0.0.0.0

Yahoo Finance futures quotes are continuous-contract proxies, not direct TopstepX
data. The app does not place orders and does not guarantee signals or targets.
