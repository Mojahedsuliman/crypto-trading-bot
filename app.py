import telebot
import time
import pandas as pd
import numpy as np
from datetime import datetime
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
import threading
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# ========== إعدادات البوت ==========
TOKEN = "8770804155:AAGisTnHi_91GiYPOV5m2Hg8x_-h1n4Gy4g"
CHAT_ID = 7779443498
API_KEY = "IUQrqT56HjEDamqO7lkroFcF9YVUIUP68uuXIt1UfUrl8fmbXcPdrs1wkYlYZpO6"
SECRET_KEY = "jU1MH7kApbvZVnYKvkxfsiY22Zqfnncv9C1D7bQ7mk9YvyOZbZk6dEkF7Kyh33H3"
# ==========================================

bot = telebot.TeleBot(TOKEN)
client = Client(API_KEY, SECRET_KEY)

# قائمة العملات
MAIN_COINS = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE', 'MATIC', 'DOT', 'LINK', 
              'AVAX', 'UNI', 'ATOM', 'ALGO', 'VET', 'FIL', 'NEAR', 'APT', 'ARB', 'OP']

class ProfessionalAnalyzer:
    def __init__(self):
        self.model_buy = None
        self.model_sell = None
        self.scaler = StandardScaler()
        self.model_buy_file = "pro_model_buy.pkl"
        self.model_sell_file = "pro_model_sell.pkl"
        self.scaler_file = "scaler.pkl"
        self.load_models()

    def calculate_rsi(self, prices, period=14):
        """مؤشر القوة النسبية RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """مؤشر MACD"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    def calculate_bollinger(self, prices, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower, sma

    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """مؤشر Stochastic RSI"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d

    def calculate_atr(self, high, low, close, period=14):
        """Average True Range"""
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_obv(self, close, volume):
        """On-Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    def get_all_indicators(self, symbol):
        """جلب جميع المؤشرات الفنية للعملة"""
        try:
            klines = client.get_klines(symbol=symbol, interval='4h', limit=200)
            df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume',
                                               'close_time', 'qa_volume', 'trades', 'taker_buy_base',
                                               'taker_buy_quote', 'ignore'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            # حساب جميع المؤشرات
            rsi = self.calculate_rsi(df['close'])
            macd, macd_signal, macd_hist = self.calculate_macd(df['close'])
            bb_upper, bb_lower, bb_middle = self.calculate_bollinger(df['close'])
            stoch_k, stoch_d = self.calculate_stochastic(df['high'], df['low'], df['close'])
            atr = self.calculate_atr(df['high'], df['low'], df['close'])
            obv = self.calculate_obv(df['close'], df['volume'])
            
            # المتوسطات المتحركة
            sma_20 = df['close'].rolling(20).mean()
            sma_50 = df['close'].rolling(50).mean()
            ema_20 = df['close'].ewm(span=20).mean()
            
            # حجم التداول
            volume_sma = df['volume'].rolling(20).mean()
            volume_ratio = df['volume'] / volume_sma
            
            # التغيرات السعرية
            price_change_1h = df['close'].pct_change(4) * 100
            price_change_4h = df['close'].pct_change(16) * 100
            price_change_24h = df['close'].pct_change(24) * 100
            
            # إنشاء DataFrame بالمؤشرات
            indicators = pd.DataFrame(index=df.index)
            indicators['rsi'] = rsi
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_hist'] = macd_hist
            indicators['bb_upper'] = bb_upper
            indicators['bb_lower'] = bb_lower
            indicators['bb_middle'] = bb_middle
            indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
            indicators['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            indicators['stoch_k'] = stoch_k
            indicators['stoch_d'] = stoch_d
            indicators['atr'] = atr
            indicators['obv'] = obv
            indicators['sma_20'] = sma_20
            indicators['sma_50'] = sma_50
            indicators['ema_20'] = ema_20
            indicators['price_sma_20'] = df['close'] / sma_20
            indicators['price_sma_50'] = df['close'] / sma_50
            indicators['volume_ratio'] = volume_ratio
            indicators['price_change_1h'] = price_change_1h
            indicators['price_change_4h'] = price_change_4h
            indicators['price_change_24h'] = price_change_24h
            indicators['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            indicators['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # إزالة القيم المفقودة
            indicators = indicators.dropna()
            
            # آخر القيم للتحليل
            latest = {}
            for col in indicators.columns:
                if len(indicators) > 0:
                    latest[col] = indicators[col].iloc[-1]
            
            latest['current_price'] = df['close'].iloc[-1]
            latest['high_24h'] = df['high'].iloc[-24:].max() if len(df) >= 24 else df['high'].iloc[-1]
            latest['low_24h'] = df['low'].iloc[-24:].min() if len(df) >= 24 else df['low'].iloc[-1]
            latest['volume'] = df['volume'].iloc[-1]
            
            return indicators, latest, df
            
        except Exception as e:
            print(f"خطأ في جلب المؤشرات: {e}")
            return None, None, None

    def create_labels(self, df, future_bars=8, profit_target=0.025, loss_limit=0.015):
        """إنشاء تسميات للتعلم الآلي"""
        future_high = df['high'].shift(-future_bars).rolling(future_bars).max()
        future_low = df['low'].shift(-future_bars).rolling(future_bars).min()
        current_price = df['close']
        
        potential_profit = (future_high - current_price) / current_price
        potential_loss = (current_price - future_low) / current_price
        
        buy_signal = (potential_profit > profit_target) & (potential_loss < loss_limit)
        sell_signal = (potential_loss > profit_target) & (potential_profit < loss_limit)
        
        return buy_signal.astype(int), sell_signal.astype(int)

    def train_models(self):
        """تدريب نماذج التعلم الآلي"""
        print("🧠 جاري تدريب نماذج الذكاء الاصطناعي...")
        print("قد يستغرق 2-3 دقائق...")
        
        all_features = []
        all_labels_buy = []
        all_labels_sell = []
        
        train_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT', 'DOGEUSDT']
        
        for symbol in train_symbols:
            print(f"📊 جلب بيانات {symbol}...")
            indicators, _, df = self.get_all_indicators(symbol)
            
            if indicators is not None and len(indicators) > 100:
                buy_labels, sell_labels = self.create_labels(df)
                
                # محاذاة البيانات
                min_len = min(len(indicators), len(buy_labels), len(sell_labels))
                features = indicators.iloc[:min_len]
                buy_lbl = buy_labels.iloc[:min_len]
                sell_lbl = sell_labels.iloc[:min_len]
                
                all_features.append(features)
                all_labels_buy.append(buy_lbl)
                all_labels_sell.append(sell_lbl)
                
                print(f"   ✅ {symbol}: {len(features)} نقطة بيانات")
                time.sleep(1)
        
        if all_features:
            X = pd.concat(all_features).fillna(0)
            y_buy = pd.concat(all_labels_buy).fillna(0)
            y_sell = pd.concat(all_labels_sell).fillna(0)
            
            # توحيد البيانات
            X_scaled = self.scaler.fit_transform(X)
            
            # تدريب النماذج
            print("🔄 تدريب نموذج الشراء...")
            self.model_buy = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.model_buy.fit(X_scaled, y_buy)
            
            print("🔄 تدريب نموذج البيع...")
            self.model_sell = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.model_sell.fit(X_scaled, y_sell)
            
            # حساب الدقة
            buy_accuracy = self.model_buy.score(X_scaled, y_buy)
            sell_accuracy = self.model_sell.score(X_scaled, y_sell)
            
            print(f"\n📈 دقة نموذج الشراء: {buy_accuracy:.2%}")
            print(f"📉 دقة نموذج البيع: {sell_accuracy:.2%}")
            
            self.save_models()
            return True
        return False

    def predict(self, symbol):
        """التنبؤ بإشارة شراء أو بيع"""
        if self.model_buy is None or self.model_sell is None:
            return "WAIT", 0.5, None, None
        
        indicators, latest, _ = self.get_all_indicators(symbol)
        if indicators is None or len(indicators) == 0:
            return "WAIT", 0.5, None, latest
        
        last_features = indicators.iloc[-1:].fillna(0)
        
        try:
            # توحيد البيانات بنفس مقياس التدريب
            last_scaled = self.scaler.transform(last_features)
            
            pred_buy = self.model_buy.predict(last_scaled)[0]
            pred_sell = self.model_sell.predict(last_scaled)[0]
            prob_buy = self.model_buy.predict_proba(last_scaled)[0].max()
            prob_sell = self.model_sell.predict_proba(last_scaled)[0].max()
            
            if pred_buy == 1 and prob_buy > 0.55:
                return "BUY", prob_buy, indicators, latest
            elif pred_sell == 1 and prob_sell > 0.55:
                return "SELL", prob_sell, indicators, latest
                
        except Exception as e:
            print(f"خطأ في التنبؤ: {e}")
        
        return "WAIT", 0.5, None, latest

    def save_models(self):
        if self.model_buy:
            with open(self.model_buy_file, 'wb') as f:
                pickle.dump(self.model_buy, f)
        if self.model_sell:
            with open(self.model_sell_file, 'wb') as f:
                pickle.dump(self.model_sell, f)
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        print("💾 تم حفظ النماذج")

    def load_models(self):
        loaded = False
        if os.path.exists(self.model_buy_file) and os.path.exists(self.scaler_file):
            with open(self.model_buy_file, 'rb') as f:
                self.model_buy = pickle.load(f)
            with open(self.model_sell_file, 'rb') as f:
                self.model_sell = pickle.load(f)
            with open(self.scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            loaded = True
            print("✅ تم تحميل النماذج المحفوظة")
        return loaded

analyzer = ProfessionalAnalyzer()

if not analyzer.load_models():
    print("📚 لا توجد نماذج محفوظة - بدء التدريب...")
    analyzer.train_models()

def get_detailed_reasons(direction, latest, indicators=None):
    """تحليل مفصل لأسباب الإشارة"""
    reasons = []
    
    if direction == "BUY":
        # أسباب الشراء
        if latest.get('rsi', 50) < 35:
            reasons.append(f"📉 RSI منخفض ({latest['rsi']:.1f}) - منطقة تشبع بيعي قوية")
        if latest.get('bb_position', 0.5) < 0.2:
            reasons.append(f"📊 السعر低于 Bollinger السفلي - السعر منخفض جدا")
        if latest.get('stoch_k', 50) < 20:
            reasons.append(f"🎯 Stochastic في منطقة تشبع بيعي ({latest['stoch_k']:.1f})")
        if latest.get('macd_hist', 0) > 0 and indicators is not None and len(indicators) > 1:
            prev_hist = indicators['macd_hist'].iloc[-2] if len(indicators) > 1 else 0
            if latest['macd_hist'] > 0 and prev_hist <= 0:
                reasons.append(f"📈 تقاطع MACD إيجابي - زخم صاعد جديد")
        if latest.get('volume_ratio', 1) > 1.5:
            reasons.append(f"📊 حجم تداول مرتفع ({latest['volume_ratio']:.1f}x المعدل)")
        if latest.get('price_change_24h', 0) < -5:
            reasons.append(f"🔄 هبوط حاد {latest['price_change_24h']:.1f}% - فرصة ارتداد")
        if latest.get('price_sma_20', 1) < 0.95:
            reasons.append(f"📉 السعر أقل 5% من SMA20 - فرصة شراء")
            
    else:
        # أسباب البيع
        if latest.get('rsi', 50) > 65:
            reasons.append(f"📈 RSI مرتفع ({latest['rsi']:.1f}) - منطقة تشبع شرائي قوية")
        if latest.get('bb_position', 0.5) > 0.8:
            reasons.append(f"📊 السعر فوق Bollinger العلوي - السعر مرتفع جدا")
        if latest.get('stoch_k', 50) > 80:
            reasons.append(f"🎯 Stochastic في منطقة تشبع شرائي ({latest['stoch_k']:.1f})")
        if latest.get('macd_hist', 0) < 0 and indicators is not None and len(indicators) > 1:
            prev_hist = indicators['macd_hist'].iloc[-2] if len(indicators) > 1 else 0
            if latest['macd_hist'] < 0 and prev_hist >= 0:
                reasons.append(f"📉 تقاطع MACD سلبي - زخم هابط جديد")
        if latest.get('volume_ratio', 1) > 1.5:
            reasons.append(f"📊 حجم تداول مرتفع ({latest['volume_ratio']:.1f}x المعدل)")
        if latest.get('price_change_24h', 0) > 5:
            reasons.append(f"🔄 صعود حاد {latest['price_change_24h']:.1f}% - فرصة جني أرباح")
        if latest.get('price_sma_20', 1) > 1.05:
            reasons.append(f"📈 السعر أعلى 5% من SMA20 - فرصة بيع")
    
    if not reasons:
        reasons.append("🤖 التحليل الآلي يشير إلى فرصة محتملة")
    
    return reasons

def calculate_advanced_levels(price, direction, latest):
    """حساب نقاط الدخول والخروج المتقدمة"""
    # حساب ATR من المؤشرات
    atr = latest.get('atr', price * 0.02)
    if pd.isna(atr) or atr == 0:
        atr = price * 0.02
    
    if direction == "BUY":
        # نقاط الشراء
        stop_loss = price - (atr * 1.5)
        tp1 = price + (atr * 1.5)
        tp2 = price + (atr * 2.5)
        tp3 = price + (atr * 4)
        
        # نقاط دخول إضافية (DCA)
        entry_2 = price - (atr * 0.5)
        entry_3 = price - (atr * 1)
    else:
        # نقاط البيع
        stop_loss = price + (atr * 1.5)
        tp1 = price - (atr * 1.5)
        tp2 = price - (atr * 2.5)
        tp3 = price - (atr * 4)
        
        entry_2 = price + (atr * 0.5)
        entry_3 = price + (atr * 1)
    
    return {
        'entry': round(price, 6),
        'entry_2': round(entry_2, 6),
        'entry_3': round(entry_3, 6),
        'stop_loss': round(stop_loss, 6),
        'tp1': round(tp1, 6),
        'tp2': round(tp2, 6),
        'tp3': round(tp3, 6),
        'atr': round(atr, 6),
        'risk_reward': "1:4"
    }

def get_all_opportunities():
    """جلب أفضل فرص الشراء والبيع"""
    buy_opportunities = []
    sell_opportunities = []
    
    print(f"🔍 جاري تحليل {len(MAIN_COINS)} عملة بـ 10+ مؤشرات...")
    
    for coin in MAIN_COINS:
        try:
            direction, confidence, indicators, latest = analyzer.predict(f"{coin}USDT")
            
            if latest and direction != "WAIT" and confidence > 0.55:
                levels = calculate_advanced_levels(latest['current_price'], direction, latest)
                reasons = get_detailed_reasons(direction, latest, indicators)
                
                opp = {
                    'symbol': coin,
                    'direction': direction,
                    'price': latest['current_price'],
                    'confidence': confidence,
                    'rsi': latest.get('rsi', 50),
                    'macd': latest.get('macd_hist', 0),
                    'stoch': latest.get('stoch_k', 50),
                    'volume_ratio': latest.get('volume_ratio', 1),
                    'bb_position': latest.get('bb_position', 0.5),
                    'reasons': reasons,
                    **levels
                }
                
                if direction == "BUY":
                    buy_opportunities.append(opp)
                    print(f"🟢 {coin}: شراء (ثقة: {confidence:.0%}, RSI: {latest.get('rsi', 50):.1f})")
                else:
                    sell_opportunities.append(opp)
                    print(f"🔴 {coin}: بيع (ثقة: {confidence:.0%}, RSI: {latest.get('rsi', 50):.1f})")
            
            time.sleep(0.3)
            
        except Exception as e:
            print(f"⚠️ خطأ في {coin}: {e}")
            continue
        
        if len(buy_opportunities) >= 15 and len(sell_opportunities) >= 15:
            break
    
    buy_opportunities.sort(key=lambda x: x['confidence'], reverse=True)
    sell_opportunities.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"\n📊 النتائج: {len(buy_opportunities)} شراء, {len(sell_opportunities)} بيع")
    return buy_opportunities[:10], sell_opportunities[:10]

def format_detailed_signal(opp, list_type):
    """تنسيق الإشارة بشكل احترافي"""
    arrow = "🟢" if opp['direction'] == "BUY" else "🔴"
    dir_ar = "شراء" if opp['direction'] == "BUY" else "بيع"
    
    # تحليل إضافي
    macd_status = "📈 إيجابي" if opp.get('macd', 0) > 0 else "📉 سلبي"
    stoch_status = "🔥 تشبع شرائي" if opp.get('stoch', 50) > 80 else "❄️ تشبع بيعي" if opp.get('stoch', 50) < 20 else "⚖️ محايد"
    bb_status = "📊 فوق العلوي" if opp.get('bb_position', 0.5) > 0.8 else "📊 تحت السفلي" if opp.get('bb_position', 0.5) < 0.2 else "📊 في المنتصف"
    
    msg = f"""
📊 *تحليل {opp['symbol']}* 🤖

{arrow} *التوصية:* {dir_ar}
🎯 *الثقة:* `{opp['confidence']:.0%}`
💰 *السعر الحالي:* `${opp['price']:.6f}`

📈 *المؤشرات الفنية:*
• RSI: `{opp['rsi']:.1f}` (30-70 طبيعي)
• MACD: {macd_status}
• Stochastic: {stoch_status}
• Bollinger: {bb_status}
• حجم التداول: `{opp['volume_ratio']:.1f}x` المعدل

📝 *أسباب التوصية:*
"""
    for r in opp['reasons'][:4]:
        msg += f"  {r}\n"
    
    msg += f"""
📐 *نقاط الدخول والخروج:*
• 🚪 *الدخول الرئيسي:* `${opp['entry']:.6f}`
• 📍 *دخول ثانوي (DCA):* `${opp['entry_2']:.6f}`
• 📍 *دخول ثالث (DCA):* `${opp['entry_3']:.6f}`
• 🛑 *وقف الخسارة:* `${opp['stop_loss']:.6f}`
• 🎯 *الهدف 1:* `${opp['tp1']:.6f}` 🟢
• 🎯 *الهدف 2:* `${opp['tp2']:.6f}` 🟡
• 🎯 *الهدف 3:* `${opp['tp3']:.6f}` 🔴
• 📊 *ATR (التقلب):* `${opp['atr']:.6f}`

⚠️ *تنبيه:* هذا تحليل آلي، طبق إدارة المخاطر بنفسك
"""
    return msg

def create_back_button(list_type, page):
    markup = InlineKeyboardMarkup()
    back_btn = InlineKeyboardButton("🔙 رجوع إلى القائمة", callback_data=f"back_{list_type}_{page}")
    markup.add(back_btn)
    return markup

def create_main_menu(list_type, opportunities, page):
    markup = InlineKeyboardMarkup(row_width=2)
    
    start_idx = (page - 1) * 10
    end_idx = min(start_idx + 10, len(opportunities))
    
    for i, opp in enumerate(opportunities[start_idx:end_idx], start_idx + 1):
        emoji = "🟢" if list_type == "buy" else "🔴"
        btn = InlineKeyboardButton(f"{emoji} #{i} {opp['symbol']} ({opp['confidence']:.0%})", 
                                   callback_data=f"detail_{list_type}_{i}")
        markup.add(btn)
    
    nav_buttons = []
    if page > 1:
        nav_buttons.append(InlineKeyboardButton("⬅️ السابق", callback_data=f"prev_{list_type}_{page-1}"))
    if end_idx < len(opportunities):
        nav_buttons.append(InlineKeyboardButton("التالي ➡️", callback_data=f"next_{list_type}_{page+1}"))
    
    if nav_buttons:
        markup.row(*nav_buttons)
    
    return markup

bot_data = {}

@bot.message_handler(commands=['start'])
def send_welcome(message):
    msg = f"""
🚀 *بوت التداول الاحترافي* 🤖

📊 *المؤشرات المستخدمة:*
• RSI, MACD, Bollinger Bands
• Stochastic, ATR, OBV
• المتوسطات المتحركة (SMA/EMA)
• تحليل الحجم والتقلب

📋 *الأوامر:*
• `/daily` - أفضل 10 شراء + 10 بيع
• `/buy` - أفضل 10 فرص شراء
• `/sell` - أفضل 10 فرص بيع
• `/coin BTC` - تحليل عملة

✅ تم تحليل {len(MAIN_COINS)} عملة بـ 15+ مؤشر
"""
    bot.reply_to(message, msg, parse_mode='Markdown')

@bot.message_handler(commands=['buy'])
def send_buy(message):
    bot.reply_to(message, "📊 جاري البحث عن فرص شراء... ⏳")
    
    def analyze_and_send():
        try:
            buy_opps, _ = get_all_opportunities()
            if not buy_opps:
                bot.send_message(message.chat.id, "⚠️ لا توجد فرص شراء حالياً")
                return
            
            bot_data[message.chat.id] = {
                'buy': buy_opps[:20],
                'sell': [],
                'buy_page': 1,
                'sell_page': 1
            }
            markup = create_main_menu('buy', buy_opps[:20], 1)
            bot.send_message(message.chat.id, "🟢 *أفضل فرص الشراء:*", reply_markup=markup, parse_mode='Markdown')
        except Exception as e:
            bot.send_message(message.chat.id, f"❌ خطأ: {e}")
    
    threading.Thread(target=analyze_and_send, daemon=True).start()

@bot.message_handler(commands=['sell'])
def send_sell(message):
    bot.reply_to(message, "📊 جاري البحث عن فرص بيع... ⏳")
    
    def analyze_and_send():
        try:
            _, sell_opps = get_all_opportunities()
            if not sell_opps:
                bot.send_message(message.chat.id, "⚠️ لا توجد فرص بيع حالياً")
                return
            
            bot_data[message.chat.id] = {
                'buy': [],
                'sell': sell_opps[:20],
                'buy_page': 1,
                'sell_page': 1
            }
            markup = create_main_menu('sell', sell_opps[:20], 1)
            bot.send_message(message.chat.id, "🔴 *أفضل فرص البيع:*", reply_markup=markup, parse_mode='Markdown')
        except Exception as e:
            bot.send_message(message.chat.id, f"❌ خطأ: {e}")
    
    threading.Thread(target=analyze_and_send, daemon=True).start()

@bot.message_handler(commands=['daily'])
def send_daily(message):
    bot.reply_to(message, "📊 جاري تحليل السوق بـ 15+ مؤشر... ⏳")
    
    def analyze_and_send():
        try:
            buy_opps, sell_opps = get_all_opportunities()
            
            bot_data[message.chat.id] = {
                'buy': buy_opps[:20],
                'sell': sell_opps[:20],
                'buy_page': 1,
                'sell_page': 1
            }
            
            if buy_opps:
                markup = create_main_menu('buy', buy_opps[:20], 1)
                bot.send_message(message.chat.id, "🟢 *فرص الشراء:*", reply_markup=markup, parse_mode='Markdown')
                time.sleep(0.5)
            
            if sell_opps:
                markup = create_main_menu('sell', sell_opps[:20], 1)
                bot.send_message(message.chat.id, "🔴 *فرص البيع:*", reply_markup=markup, parse_mode='Markdown')
            
            if not buy_opps and not sell_opps:
                bot.send_message(message.chat.id, "⚠️ لا توجد فرص تداول واضحة حالياً. حاول مرة أخرى لاحقاً.")
        except Exception as e:
            bot.send_message(message.chat.id, f"❌ خطأ: {e}")
    
    threading.Thread(target=analyze_and_send, daemon=True).start()

@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    try:
        data = call.data
        chat_id = call.message.chat.id
        
        if chat_id not in bot_data:
            bot.answer_callback_query(call.id, "⚠️ انتهت الصلاحية، أعد /daily")
            return
        
        if data.startswith('back_'):
            parts = data.split('_')
            list_type = parts[1]
            page = int(parts[2])
            
            opportunities = bot_data[chat_id].get(list_type, [])
            if opportunities:
                markup = create_main_menu(list_type, opportunities, page)
                title = "🟢 فرص الشراء:" if list_type == "buy" else "🔴 فرص البيع:"
                bot.edit_message_text(title, chat_id, call.message.message_id, 
                                     reply_markup=markup, parse_mode='Markdown')
        
        elif data.startswith('next_') or data.startswith('prev_'):
            parts = data.split('_')
            action = parts[0]
            list_type = parts[1]
            new_page = int(parts[2])
            
            opportunities = bot_data[chat_id].get(list_type, [])
            if opportunities:
                if list_type == 'buy':
                    bot_data[chat_id]['buy_page'] = new_page
                else:
                    bot_data[chat_id]['sell_page'] = new_page
                
                markup = create_main_menu(list_type, opportunities, new_page)
                title = "🟢 فرص الشراء:" if list_type == "buy" else "🔴 فرص البيع:"
                bot.edit_message_text(title, chat_id, call.message.message_id,
                                     reply_markup=markup, parse_mode='Markdown')
        
        elif data.startswith('detail_'):
            parts = data.split('_')
            list_type = parts[1]
            idx = int(parts[2]) - 1
            
            opportunities = bot_data[chat_id].get(list_type, [])
            if idx < len(opportunities):
                opp = opportunities[idx]
                page = bot_data[chat_id].get(f'{list_type}_page', 1)
                msg = format_detailed_signal(opp, list_type)
                markup = create_back_button(list_type, page)
                bot.edit_message_text(msg, chat_id, call.message.message_id,
                                     reply_markup=markup, parse_mode='Markdown')
        
        bot.answer_callback_query(call.id, "✅ تم")
        
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ خطأ")

@bot.message_handler(commands=['coin'])
def analyze_coin(message):
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "❌ مثال: /coin BTC")
            return
        
        symbol = parts[1].upper()
        bot.reply_to(message, f"📊 جاري تحليل {symbol} بـ 15+ مؤشر...")
        
        direction, confidence, indicators, latest = analyzer.predict(f"{symbol}USDT")
        
        if direction == "WAIT" or not latest:
            msg = f"📊 *{symbol}*\n⚠️ لا توجد إشارة واضحة حالياً"
        else:
            levels = calculate_advanced_levels(latest['current_price'], direction, latest)
            reasons = get_detailed_reasons(direction, latest, indicators)
            arrow = "🟢" if direction == "BUY" else "🔴"
            dir_ar = "شراء" if direction == "BUY" else "بيع"
            
            msg = f"""
📊 *تحليل {symbol}* {arrow}

💰 *السعر:* `${latest['current_price']:.6f}`
⚡ *التوصية:* {dir_ar} | ثقة: `{confidence:.0%}`

📈 *المؤشرات الفنية:*
• RSI: `{latest.get('rsi', 50):.1f}`
• MACD Hist: `{latest.get('macd_hist', 0):.4f}`
• Stochastic K: `{latest.get('stoch_k', 50):.1f}`
• حجم التداول: `{latest.get('volume_ratio', 1):.1f}x`
• التغير 24س: `{latest.get('price_change_24h', 0):+.2f}%`

📝 *أسباب التوصية:*
"""
            for r in reasons[:3]:
                msg += f"  {r}\n"
            
            msg += f"""
📐 *نقاط الدخول والخروج:*
• 🚪 *الدخول:* `${levels['entry']:.6f}`
• 🛑 *وقف الخسارة:* `${levels['stop_loss']:.6f}`
• 🎯 *TP1:* `${levels['tp1']:.6f}`
• 🎯 *TP2:* `${levels['tp2']:.6f}`
• 🎯 *TP3:* `${levels['tp3']:.6f}`

⚠️ للإشارة فقط - ليس نصيحة مالية
"""
        bot.send_message(message.chat.id, msg, parse_mode='Markdown')
        
    except Exception as e:
        bot.reply_to(message, f"❌ خطأ: {e}")

# ========== التشغيل ==========
print("=" * 60)
print("🚀 بوت التداول الاحترافي يعمل!")
print("📊 يستخدم 15+ مؤشر فني (RSI, MACD, Bollinger, Stochastic, ATR...)")
print("🧠 نماذج تعلم آلي (Random Forest)")
print("=" * 60)

bot.send_message(CHAT_ID, "🚀 *بوت التداول الاحترافي بدأ العمل!*\n\n📊 *المؤشرات:* RSI, MACD, Bollinger, Stochastic, ATR, OBV\n🧠 *التقنية:* Random Forest (تعلم آلي)\n\n💡 أرسل `/daily` للحصول على أفضل الفرص", parse_mode='Markdown')

bot.infinity_polling()
# ========== للتشغيل على Render ==========
from flask import Flask
import threading

flask_app = Flask(__name__)

@flask_app.route('/')
def health_check():
    return "Bot is alive!", 200

def run_flask():
    flask_app.run(host='0.0.0.0', port=10000)

# تشغيل Flask في خلفية
threading.Thread(target=run_flask, daemon=True).start()
# ==========================================