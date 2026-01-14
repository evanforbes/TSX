#!/usr/bin/env python3
"""
TSX Scanner - Telegram Bot
Send alerts and run scans via Telegram
"""

import os
import asyncio
from datetime import datetime
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

from scanner import get_tsx_symbols, fetch_stock_data, passes_filters, analyze_stock, classify_signal_tier

# Get bot token from environment variable
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "📊 *TSX Stock Scanner Bot*\n\n"
        "Commands:\n"
        "/scan - Run full market scan\n"
        "/gold - Show Gold Standard signals only\n"
        "/silver - Show Silver Standard signals only\n"
        "/check SYMBOL - Check a specific stock\n"
        "/help - Show this message\n\n"
        "I'll also send you alerts when Gold/Silver signals are found!",
        parse_mode='Markdown'
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    await start(update, context)


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /scan command - run full market scan"""
    await update.message.reply_text("🔍 Starting TSX scan... This may take a few minutes.")

    results = await run_scan()

    if not results['signals']:
        await update.message.reply_text("No signals found in this scan.")
        return

    # Send summary
    msg = format_scan_results(results)
    await update.message.reply_text(msg, parse_mode='Markdown')


async def gold_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /gold command - show gold standard only"""
    await update.message.reply_text("🔍 Scanning for Gold Standard signals...")

    results = await run_scan()

    gold_buys = [r for r in results['signals'] if r['tier'].get('tier_buy') == 'GOLD']
    gold_sells = [r for r in results['signals'] if r['tier'].get('tier_sell') == 'GOLD']

    if not gold_buys and not gold_sells:
        await update.message.reply_text("No Gold Standard signals found right now.")
        return

    msg = "🥇 *GOLD STANDARD SIGNALS*\n\n"

    if gold_buys:
        msg += "*BUY:*\n"
        for r in gold_buys:
            msg += f"• *{r['symbol']}.TO* - ${r['price']:.2f}\n"
            for ind, desc in r['tier']['buy_indicators']:
                msg += f"  _{ind}: {desc[:50]}_\n"

    if gold_sells:
        msg += "\n*SELL:*\n"
        for r in gold_sells:
            msg += f"• *{r['symbol']}.TO* - ${r['price']:.2f}\n"
            for ind, desc in r['tier']['sell_indicators']:
                msg += f"  _{ind}: {desc[:50]}_\n"

    await update.message.reply_text(msg, parse_mode='Markdown')


async def silver_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /silver command - show silver standard only"""
    await update.message.reply_text("🔍 Scanning for Silver Standard signals...")

    results = await run_scan()

    silver_buys = [r for r in results['signals'] if r['tier'].get('tier_buy') == 'SILVER']
    silver_sells = [r for r in results['signals'] if r['tier'].get('tier_sell') == 'SILVER']

    if not silver_buys and not silver_sells:
        await update.message.reply_text("No Silver Standard signals found right now.")
        return

    msg = "🥈 *SILVER STANDARD SIGNALS*\n\n"

    if silver_buys:
        msg += "*BUY:*\n"
        for r in silver_buys:
            msg += f"• *{r['symbol']}.TO* - ${r['price']:.2f}\n"
            for ind, desc in r['tier']['buy_indicators']:
                msg += f"  _{ind}: {desc[:50]}_\n"

    if silver_sells:
        msg += "\n*SELL:*\n"
        for r in silver_sells:
            msg += f"• *{r['symbol']}.TO* - ${r['price']:.2f}\n"
            for ind, desc in r['tier']['sell_indicators']:
                msg += f"  _{ind}: {desc[:50]}_\n"

    await update.message.reply_text(msg, parse_mode='Markdown')


async def check_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /check SYMBOL command"""
    if not context.args:
        await update.message.reply_text("Usage: /check SYMBOL\nExample: /check RY")
        return

    symbol = context.args[0].upper().replace('.TO', '')
    await update.message.reply_text(f"🔍 Checking {symbol}.TO...")

    df = fetch_stock_data(symbol)
    if df is None:
        await update.message.reply_text(f"❌ Could not fetch data for {symbol}.TO")
        return

    passes, reason, price, volume = passes_filters(df)
    result = analyze_stock(symbol, df)
    tier = classify_signal_tier(result)

    # Format response
    msg = f"📈 *{symbol}.TO*\n"
    msg += f"Price: ${price:.2f}\n"
    msg += f"Volume: {volume:,.0f}\n\n"

    # Tier badge
    if tier['tier_buy'] == 'GOLD':
        msg += "🥇 *GOLD STANDARD BUY*\n\n"
    elif tier['tier_sell'] == 'GOLD':
        msg += "🥇 *GOLD STANDARD SELL*\n\n"
    elif tier['tier_buy'] == 'SILVER':
        msg += "🥈 *SILVER STANDARD BUY*\n\n"
    elif tier['tier_sell'] == 'SILVER':
        msg += "🥈 *SILVER STANDARD SELL*\n\n"

    # Indicators
    msg += "*Indicators:*\n"
    msg += f"• RSI: {result['rsi']['rsi_value']:.1f}\n"
    msg += f"• MACD Hist: {result['macd']['histogram_value']:.4f}\n"
    msg += f"• SlowSto %K: {result['stochastic']['k_value']:.1f}\n"
    msg += f"• P&F: {'Triple Top Breakout!' if result['pf']['triple_top_breakout'] else result['pf']['current_direction'] or 'N/A'}\n\n"

    # Active signals
    if result['signals']:
        msg += "*Active Signals:*\n"
        for indicator, signal, desc in result['signals']:
            emoji = "🟢" if signal in ('BUY', 'STRONG_BUY') else "🔴"
            msg += f"{emoji} {indicator}: {desc[:60]}\n"
    else:
        msg += "_No active signals_"

    await update.message.reply_text(msg, parse_mode='Markdown')


async def run_scan():
    """Run the scanner and return categorized results"""
    symbols = get_tsx_symbols()
    signals = []

    for symbol in symbols:
        df = fetch_stock_data(symbol)
        if df is None:
            continue

        passes, reason, price, volume = passes_filters(df)
        if not passes:
            continue

        result = analyze_stock(symbol, df)
        if result['signals']:
            tier = classify_signal_tier(result)
            result['tier'] = tier
            signals.append(result)

    return {
        'timestamp': datetime.now().isoformat(),
        'total_scanned': len(symbols),
        'signals': signals
    }


def format_scan_results(results):
    """Format scan results for Telegram message"""
    signals = results['signals']

    gold_buys = [r for r in signals if r['tier'].get('tier_buy') == 'GOLD']
    gold_sells = [r for r in signals if r['tier'].get('tier_sell') == 'GOLD']
    silver_buys = [r for r in signals if r['tier'].get('tier_buy') == 'SILVER']
    silver_sells = [r for r in signals if r['tier'].get('tier_sell') == 'SILVER']

    msg = "📊 *TSX SCAN COMPLETE*\n"
    msg += f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"

    msg += "*Summary:*\n"
    msg += f"🥇 Gold Buys: {len(gold_buys)}\n"
    msg += f"🥇 Gold Sells: {len(gold_sells)}\n"
    msg += f"🥈 Silver Buys: {len(silver_buys)}\n"
    msg += f"🥈 Silver Sells: {len(silver_sells)}\n"
    msg += f"📈 Total Signals: {len(signals)}\n\n"

    # Show gold/silver details
    if gold_buys:
        msg += "🥇 *GOLD BUY:*\n"
        for r in gold_buys[:5]:
            msg += f"• {r['symbol']}.TO ${r['price']:.2f}\n"

    if gold_sells:
        msg += "\n🥇 *GOLD SELL:*\n"
        for r in gold_sells[:5]:
            msg += f"• {r['symbol']}.TO ${r['price']:.2f}\n"

    if silver_buys:
        msg += "\n🥈 *SILVER BUY:*\n"
        for r in silver_buys[:5]:
            msg += f"• {r['symbol']}.TO ${r['price']:.2f}\n"

    if silver_sells:
        msg += "\n🥈 *SILVER SELL:*\n"
        for r in silver_sells[:5]:
            msg += f"• {r['symbol']}.TO ${r['price']:.2f}\n"

    return msg


async def send_alert(message: str):
    """Send an alert message to the configured chat"""
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured - skipping alert")
        return

    bot = Bot(token=BOT_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')


def run_bot():
    """Run the Telegram bot"""
    if not BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set")
        print("\nTo set up Telegram bot:")
        print("1. Message @BotFather on Telegram")
        print("2. Send /newbot and follow instructions")
        print("3. Copy the token and set: export TELEGRAM_BOT_TOKEN='your-token'")
        print("4. Message your bot, then get chat ID from:")
        print("   https://api.telegram.org/bot<TOKEN>/getUpdates")
        print("5. Set: export TELEGRAM_CHAT_ID='your-chat-id'")
        return

    print("Starting Telegram bot...")
    app = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("gold", gold_command))
    app.add_handler(CommandHandler("silver", silver_command))
    app.add_handler(CommandHandler("check", check_command))

    print("Bot is running! Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    run_bot()
