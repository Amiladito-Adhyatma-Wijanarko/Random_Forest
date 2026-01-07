from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import io
import base64
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ============================================================
# LOAD MODEL, ENCODER DAN FEATURE COLUMNS
# ============================================================
MODEL_PATH = 'model_random_forestt.pkl'
ENCODER_PATH = 'motif_encoderr.pkl'
FEATURE_COLS_PATH = 'feature_columnss.pkl'

loaded_model = None
motif_uniques = None
feature_columns = None

try:
    loaded_model = joblib.load(MODEL_PATH)
    motif_uniques = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLS_PATH)
    print(f"✓ Model berhasil dimuat: {MODEL_PATH}")
    print(f"✓ Encoder berhasil dimuat: {ENCODER_PATH}")
    print(f"✓ Feature columns berhasil dimuat: {FEATURE_COLS_PATH}")
    print(f"✓ Jumlah fitur: {len(feature_columns)}")
except Exception as e:
    print(f"✗ Error loading files: {e}")

# ============================================================
# GLOBAL VARIABLES
# ============================================================
df_global = None
df_features = None
X_train = None
X_val = None
X_test = None
y_train = None
y_val = None
y_test = None
total_records = 0
current_akurasi = 0.0

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def create_features(df):
    df = df.copy()
    df = df.sort_values(['Motif_Encoded', 'Minggu']).reset_index(drop=True)
    
    df['Lag_1'] = df.groupby('Motif_Encoded')['Jumlah'].shift(1)
    df['Lag_2'] = df.groupby('Motif_Encoded')['Jumlah'].shift(2)
    df['Lag_3'] = df.groupby('Motif_Encoded')['Jumlah'].shift(3)
    
    df['Rolling_Mean_3'] = df.groupby('Motif_Encoded')['Jumlah'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['Rolling_Std_3'] = df.groupby('Motif_Encoded')['Jumlah'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std()
    )
    
    df['Trend'] = df.groupby('Motif_Encoded')['Jumlah'].diff()
    
    motif_stats = df.groupby('Motif_Encoded')['Jumlah'].agg(['mean', 'std', 'min', 'max'])
    motif_stats.columns = ['Motif_Mean', 'Motif_Std', 'Motif_Min', 'Motif_Max']
    df = df.merge(motif_stats, on='Motif_Encoded', how='left')
    
    df['Minggu_Sin'] = np.sin(2 * np.pi * df['Minggu'] / 52)
    df['Minggu_Cos'] = np.cos(2 * np.pi * df['Minggu'] / 52)
    
    df = df.fillna(0)
    
    return df

def create_plot(y_test, y_pred):
    plt.figure(figsize=(14, 6))
    
    df_plot = pd.DataFrame({
        'Actual': y_test.values if hasattr(y_test, 'values') else y_test,
        'Predicted': y_pred
    })
    df_plot = df_plot.sort_values('Actual').reset_index(drop=True)
    
    plt.plot(df_plot.index, df_plot['Actual'], 
             color='#06b6d4', linewidth=2, label='Penjualan Aktual', 
             marker='o', markersize=4, alpha=0.7)
    plt.plot(df_plot.index, df_plot['Predicted'], 
             color='#3b82f6', linewidth=2, label='Prediksi Sistem', 
             marker='o', markersize=4, alpha=0.7)
    
    plt.fill_between(df_plot.index, df_plot['Actual'], alpha=0.2, color='#06b6d4')
    plt.fill_between(df_plot.index, df_plot['Predicted'], alpha=0.2, color='#3b82f6')
    
    plt.xlabel("Periode", fontsize=12, fontweight='bold')
    plt.ylabel("Jumlah Stok (Pcs)", fontsize=12, fontweight='bold')
    plt.title("Perbandingan Aktual vs Prediksi", fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return plot_data

def create_feature_importance_plot():
    if loaded_model is None or not hasattr(loaded_model, 'feature_importances_'):
        return None
    
    try:
        importances = loaded_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        
        bars = plt.barh(importance_df['Feature'], importance_df['Importance'], 
                        color='#17a2b8', edgecolor='#0f7d8f', linewidth=1.5)
        
        for i, (feature, importance) in enumerate(zip(importance_df['Feature'], importance_df['Importance'])):
            plt.text(importance + 0.005, i, f'{importance:.3f}', 
                    va='center', fontsize=10, fontweight='bold')
        
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.title('Feature Importance (Fitur Paling Berpengaruh)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return plot_data
        
    except Exception as e:
        print(f"✗ Error membuat feature importance plot: {e}")
        return None

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def dashboard():
    global current_akurasi
    
    status_model = "Tersedia" if loaded_model is not None else "Tidak Tersedia"
    
    stats = {
        'status_model': status_model,
        'akurasi': current_akurasi,
        'data_training': total_records,
        'plot_data': None
    }
    
    if X_test is not None and y_test is not None and loaded_model is not None and len(X_test) > 0:
        try:
            y_pred_test = loaded_model.predict(X_test)
            stats['plot_data'] = create_plot(y_test, y_pred_test)
        except Exception as e:
            print(f"✗ Error generating plot: {e}")
    
    return render_template('dashboard.html', **stats)

@app.route('/upload', methods=['GET', 'POST'])
def upload_data():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Tidak ada file yang diupload'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Tidak ada file yang dipilih'}), 400
        
        if file and (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            try:
                global df_global, df_features, X_train, X_val, X_test, y_train, y_val, y_test, total_records, current_akurasi
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_extension = file.filename.rsplit('.', 1)[1].lower()
                saved_filename = f"data_{timestamp}.{file_extension}"
                
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
                file.save(filepath)
                
                df = pd.read_excel(filepath)
                df_global = df.copy()
                
                print(f"✓ Data berhasil dibaca: {len(df)} baris")
                
                minggu_cols = [c for c in df.columns if 'minggu' in c.lower()]
                
                if 'Motif' in df.columns:
                    motif_col = 'Motif'
                elif 'Nama Motif' in df.columns:
                    motif_col = 'Nama Motif'
                elif 'Kode Motif' in df.columns:
                    motif_col = 'Kode Motif'
                else:
                    return jsonify({'success': False, 'error': 'Kolom Motif tidak ditemukan'}), 400
                
                if not minggu_cols:
                    return jsonify({'success': False, 'error': 'Kolom Minggu tidak ditemukan'}), 400
                
                df_long = pd.melt(
                    df,
                    id_vars=[motif_col],
                    value_vars=minggu_cols,
                    var_name='Minggu_Str',
                    value_name='Jumlah'
                )
                
                df_long['Minggu'] = df_long['Minggu_Str'].str.extract(r'(\d+)').astype(int)
                df_long.drop(columns=['Minggu_Str'], inplace=True)
                df_long.rename(columns={motif_col: 'Motif'}, inplace=True)
                
                df_long['Motif_Encoded'], _ = pd.factorize(df_long['Motif'])
                
                df_long = df_long.sort_values(['Motif', 'Minggu']).reset_index(drop=True)
                
                print(f"✓ Membuat fitur temporal...")
                df_features = create_features(df_long)
                
                X = df_features[feature_columns]
                y = df_features['Jumlah']
                
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=0.15, random_state=42, shuffle=True
                )
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.176, random_state=42, shuffle=True
                )
                
                total_records = len(df_features)
                
                print(f"✓ Data split berhasil:")
                print(f"  - Training: {len(X_train)} records")
                print(f"  - Validation: {len(X_val)} records")
                print(f"  - Test: {len(X_test)} records")
                
                if loaded_model is not None and len(X_test) > 0:
                    try:
                        y_pred_test = loaded_model.predict(X_test)
                        r2_test = r2_score(y_test, y_pred_test)
                        current_akurasi = round(r2_test * 100, 1)
                        print(f"✓ Akurasi Model: {current_akurasi}%")
                    except Exception as e:
                        print(f"✗ Error menghitung akurasi: {e}")
                        current_akurasi = 0.0
                
                motif_count = df_long['Motif'].nunique()
                
                return jsonify({
                    'success': True,
                    'message': f'Data berhasil diupload! Total {total_records} records dengan {motif_count} motif unik. Akurasi model: {current_akurasi}%',
                    'records': total_records,
                    'filename': saved_filename,
                    'motif_count': motif_count,
                    'week_count': len(minggu_cols),
                    'akurasi': current_akurasi
                })
                
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                return jsonify({'success': False, 'error': f'Error memproses file: {str(e)}'}), 500
        
        return jsonify({'success': False, 'error': 'Format file tidak valid'}), 400
    
    return render_template('upload.html')

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi_stok():
    if request.method == 'POST':
        if loaded_model is None or motif_uniques is None:
            return jsonify({'success': False, 'error': 'Model belum dimuat'}), 400
        
        if df_features is None:
            return jsonify({'success': False, 'error': 'Silakan upload data terlebih dahulu'}), 400
        
        try:
            minggu_target = int(request.form.get('minggu', 1))
            
            if minggu_target < 1:
                return jsonify({'success': False, 'error': 'Minggu harus lebih dari 0'}), 400
            
            print(f"\n===== PREDIKSI MINGGU {minggu_target} =====")
            
            results = []
            threshold = 26
            
            for motif in motif_uniques:
                motif_idx = np.where(motif_uniques == motif)[0]
                if len(motif_idx) == 0:
                    continue
                
                motif_code = motif_idx[0]
                
                motif_data = df_features[df_features['Motif_Encoded'] == motif_code].sort_values('Minggu')
                
                if len(motif_data) >= 3:
                    lag_1 = motif_data.iloc[-1]['Jumlah']
                    lag_2 = motif_data.iloc[-2]['Jumlah'] if len(motif_data) >= 2 else 0
                    lag_3 = motif_data.iloc[-3]['Jumlah'] if len(motif_data) >= 3 else 0
                    
                    recent_values = motif_data.tail(3)['Jumlah'].values
                    rolling_mean = recent_values.mean()
                    rolling_std = recent_values.std() if len(recent_values) > 1 else 0
                    trend = lag_1 - lag_2
                else:
                    lag_1 = lag_2 = lag_3 = 0
                    rolling_mean = rolling_std = trend = 0
                
                motif_mean = motif_data['Jumlah'].mean()
                motif_std = motif_data['Jumlah'].std()
                motif_min = motif_data['Jumlah'].min()
                motif_max = motif_data['Jumlah'].max()
                
                minggu_sin = np.sin(2 * np.pi * minggu_target / 52)
                minggu_cos = np.cos(2 * np.pi * minggu_target / 52)
                
                X_input = pd.DataFrame({
                    'Minggu': [minggu_target],
                    'Motif_Encoded': [motif_code],
                    'Lag_1': [lag_1],
                    'Lag_2': [lag_2],
                    'Lag_3': [lag_3],
                    'Rolling_Mean_3': [rolling_mean],
                    'Rolling_Std_3': [rolling_std],
                    'Trend': [trend],
                    'Motif_Mean': [motif_mean],
                    'Motif_Std': [motif_std],
                    'Motif_Min': [motif_min],
                    'Motif_Max': [motif_max],
                    'Minggu_Sin': [minggu_sin],
                    'Minggu_Cos': [minggu_cos]
                })
                
                y_pred = loaded_model.predict(X_input)[0]
                prediksi_val = int(round(max(0, y_pred)))
                
                kategori = 'Tinggi' if prediksi_val >= threshold else 'Rendah'
                
                results.append({
                    'Motif': motif,
                    'Prediksi': prediksi_val,
                    'Kategori': kategori
                })
            
            total_stok = sum(r['Prediksi'] for r in results)
            
            print(f"✓ Total stok prediksi: {total_stok} pcs")
            
            return jsonify({
                'success': True,
                'minggu': minggu_target,
                'total_stok': int(total_stok),
                'results': results
            })
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return jsonify({'success': False, 'error': f'Error: {str(e)}'}), 500
    
    return render_template('prediksi.html')

@app.route('/evaluasi')
def evaluasi_model():
    if loaded_model is None:
        return render_template('evaluasi.html', error='Model belum dimuat')
    
    if X_test is None or y_test is None:
        return render_template('evaluasi.html', error='Silakan upload data terlebih dahulu')
    
    try:
        y_train_pred = loaded_model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = calculate_mape(y_train, y_train_pred)
        
        y_val_pred = loaded_model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        val_mape = calculate_mape(y_val, y_val_pred)
        
        y_test_pred = loaded_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mape = calculate_mape(y_test, y_test_pred)
        
        metrics = {
            'train_rmse': round(train_rmse, 2),
            'train_mae': round(train_mae, 2),
            'train_r2': round(train_r2, 4),
            'train_r2_persen': round(train_r2 * 100, 2),
            'train_mape': round(train_mape, 2),
            'val_rmse': round(val_rmse, 2),
            'val_mae': round(val_mae, 2),
            'val_r2': round(val_r2, 4),
            'val_r2_persen': round(val_r2 * 100, 2),
            'val_mape': round(val_mape, 2),
            'test_rmse': round(test_rmse, 2),
            'test_mae': round(test_mae, 2),
            'test_r2': round(test_r2, 4),
            'test_r2_persen': round(test_r2 * 100, 2),
            'test_mape': round(test_mape, 2)
        }
        
        plot_data = create_feature_importance_plot()
        
        return render_template('evaluasi.html', metrics=metrics, plot_data=plot_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('evaluasi.html', error=f'Error evaluasi: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)