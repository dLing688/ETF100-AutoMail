import os, smtplib, ssl, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

# === Secrets / Env ===
SENDER_EMAIL     = os.environ["SENDER_EMAIL"]
APP_PASSWORD     = os.environ["APP_PASSWORD"]
RECIPIENTS       = [x.strip() for x in os.environ["RECIPIENTS"].split(",") if x.strip()]
GOOGLE_SHEET_CSV = os.environ["GOOGLE_SHEET_CSV"]

LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "120"))
CHART_DAYS    = int(os.environ.get("CHART_DAYS", "60"))
TEST_FIRST_N  = os.environ.get("TEST_FIRST_N")
TEST_FIRST_N  = int(TEST_FIRST_N) if TEST_FIRST_N and TEST_FIRST_N.strip() else None

OUT_DIR     = os.environ.get("OUT_DIR", "etf100_outputs")
REPORT_XLSX = os.environ.get("REPORT_XLSX", "ETF100_report.xlsx")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()

def calc_dmi(df, n=14):
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    up, dn = hi.diff(), -lo.diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    tr_n = tr.rolling(n).sum()
    plus_di  = 100 * pd.Series(plus_dm, index=df.index).rolling(n).sum() / tr_n
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).sum() / tr_n
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(n).mean()
    return plus_di, minus_di, adx

def calc_indicators(df):
    out = pd.DataFrame(index=df.index)
    out[["Open","High","Low","Close","Volume"]] = df[["Open","High","Low","Close","Volume"]]
    out["MACD"] = ema(out["Close"],12) - ema(out["Close"],26)
    out["MACD_signal"] = ema(out["MACD"],9)
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]
    low14, high14 = out["Low"].rolling(14).min(), out["High"].rolling(14).max()
    out["%K"] = 100*(out["Close"]-low14)/(high14-low14)
    out["%D"] = out["%K"].rolling(3).mean()
    out["J"]  = 3*out["%D"] - 2*out["%K"]
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100/(1+rs))
    out["OBV"] = (np.sign(out["Close"].diff().fillna(0)) * out["Volume"]).fillna(0).cumsum()
    out["PSY"] = 100 * (out["Close"].diff() > 0).astype(int).rolling(12).mean()
    out["W%R"] = -100*(high14 - out["Close"])/(high14-low14)
    out["BIAS6"]  = (out["Close"]/sma(out["Close"],6)  - 1)*100
    out["BIAS12"] = (out["Close"]/sma(out["Close"],12) - 1)*100
    out["BIAS24"] = (out["Close"]/sma(out["Close"],24) - 1)*100
    tp = (out["High"] + out["Low"] + out["Close"]) / 3.0
    out["VWAP"] = (tp*out["Volume"]).cumsum()/out["Volume"].cumsum().replace(0,np.nan)
    out["+DI"], out["-DI"], out["ADX"] = calc_dmi(out, 14)
    out["MA3"], out["MA5"], out["MA10"] = sma(out["Close"],3), sma(out["Close"],5), sma(out["Close"],10)
    return out

def screen_conditions(df):
    c1 = (df["+DI"] > df["-DI"]) & (df["+DI"].shift(1) <= df["-DI"].shift(1)) & (df["ADX"] > df["ADX"].shift(1))
    c2 = (df["MACD"] > 0) & (df["MACD_signal"] > 0) & (df["MACD_hist"].shift(1) < 0) & (df["MACD_hist"] > 0)
    c3 = df["OBV"] > df["OBV"].shift(1)
    c4 = (df["J"] < 20) & (df["J"] > df["J"].shift(1))
    c5 = (df["PSY"] > df["PSY"].shift(1)) & (df["PSY"].shift(1) < 40)
    c6 = df["Volume"] > 2_000_000
    return c1,c2,c3,c4,c5,c6

def make_chart(sym, dfi, out_path, last_n=60):
    d = dfi.tail(last_n)
    fig = plt.figure(figsize=(14,9))
    gs = fig.add_gridspec(6,1,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0])
    ax1.plot(d.index,d["Close"],label="Close"); ax1.plot(d.index,d["MA3"],label="MA3"); ax1.plot(d.index,d["MA5"],label="MA5"); ax1.plot(d.index,d["MA10"],label="MA10"); ax1.plot(d.index,d["VWAP"],label="VWAP")
    ax1.grid(alpha=0.2); ax1.legend(loc="upper left"); ax1_t=ax1.twinx(); ax1_t.bar(d.index,d["Volume"],alpha=0.3,label="Volume"); ax1_t.legend(loc="upper right"); ax1.set_title(f"{sym} - Price/MA/VWAP")
    ax2 = fig.add_subplot(gs[2,0]); ax2.plot(d.index,d["MACD"],label="MACD(DIF)"); ax2.plot(d.index,d["MACD_signal"],label="Signal"); ax2.bar(d.index,d["MACD_hist"],label="Hist"); ax2.grid(alpha=0.2); ax2.legend(loc="upper left"); ax2.set_title("MACD")
    ax3 = fig.add_subplot(gs[3,0]); ax3.plot(d.index,d["+DI"],label="+DI"); ax3.plot(d.index,d["-DI"],label="-DI"); ax3.plot(d.index,d["ADX"],label="ADX"); ax3.grid(alpha=0.2); ax3.legend(loc="upper left"); ax3.set_title("DMI")
    ax4 = fig.add_subplot(gs[4,0]); ax4.plot(d.index,d["%K"],label="%K"); ax4.plot(d.index,d["%D"],label="%D"); ax4.plot(d.index,d["J"],label="J"); ax4.plot(d.index,d["RSI"],label="RSI"); ax4.grid(alpha=0.2); ax4.legend(loc="upper left"); ax4.set_title("KDJ & RSI")
    ax5 = fig.add_subplot(gs[5,0]); ax5.plot(d.index,d["OBV"],label="OBV"); ax5.plot(d.index,d["PSY"],label="PSY"); ax5.plot(d.index,d["W%R"],label="W%R"); ax5.plot(d.index,d["BIAS6"],label="BIAS6"); ax5.grid(alpha=0.2); ax5.legend(loc="upper left"); ax5.set_title("OBV / PSY / W%R / BIAS6")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

def send_email(subject, body, attachments):
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL; msg["To"] = ", ".join(RECIPIENTS); msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    for p in attachments:
        with open(p,"rb") as f:
            part = MIMEBase("application","octet-stream"); part.set_payload(f.read())
        encoders.encode_base64(part); part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(p)}"'); msg.attach(part)
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com",465,context=ctx) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENTS, msg.as_string())

def load_tickers_from_sheet(url):
    df = pd.read_csv(url)
    for col in ["symbol","ticker","代號","Symbol","Ticker"]:
        if col in df.columns:
            raw = [str(x).strip() for x in df[col].dropna()]
            break
    else:
        raw = [str(x).strip() for x in df[df.columns[0]].dropna()]
    out=[]
    for s in raw:
        if s.isdigit() and not s.endswith(".TW"): s += ".TW"
        out.append(s)
    return out

def main():
    symbols = load_tickers_from_sheet(GOOGLE_SHEET_CSV)
    if TEST_FIRST_N: symbols = symbols[:TEST_FIRST_N]
    rows=[]
    for sym in symbols:
        try:
            df = yf.download(sym, period=f"{LOOKBACK_DAYS}d", interval="1d", auto_adjust=False, progress=False)
            if df.empty or len(df)<50: print(sym,"資料不足"); continue
            dfi = calc_indicators(df)
            c1,c2,c3,c4,c5,c6 = screen_conditions(dfi)
            last = dfi.iloc[-1]
            row = {"Symbol":sym,"Date":str(dfi.index[-1].date()),"Price":float(last["Close"]),
                   "Cond1_DMI_cross_ADXup":bool(c1.iloc[-1]),"Cond2_MACD_all_pos_Osc_flip":bool(c2.iloc[-1]),
                   "Cond3_OBV_up":bool(c3.iloc[-1]),"Cond4_J_lt20_up":bool(c4.iloc[-1]),
                   "Cond5_PSY_up_prev_lt40":bool(c5.iloc[-1]),"Cond6_Volume_gt_2000_lots":bool(c6.iloc[-1])}
            row["Score"]=sum(v for k,v in row.items() if k.startswith("Cond"))
            rows.append(row)
            make_chart(sym, dfi, os.path.join(OUT_DIR, f"{sym.replace('.','_')}_chart.png"), CHART_DAYS)
            print(sym,"完成")
        except Exception as e:
            print(sym,"錯誤：",e)
    report = pd.DataFrame(rows).sort_values(["Score","Symbol"], ascending=[False,True])
    report_path = os.path.join(OUT_DIR, REPORT_XLSX)
    report.to_excel(report_path, index=False)
    print("報表：", report_path)
    print("Top 20 預覽：\n", report.head(20).to_string(index=False))
    today = dt.datetime.now().strftime("%Y-%m-%d")
    subject=f"ETF100 智勝簡訊（GitHub Actions） - {today}"
    body=f"附上 {today} 的篩選報表與圖表。"
    atts=[report_path]+[str(Path(OUT_DIR)/f) for f in os.listdir(OUT_DIR) if f.endswith(".png")]
    send_email(subject, body, atts)
    print("Email 已嘗試寄出。")

if __name__ == "__main__":
    main()
