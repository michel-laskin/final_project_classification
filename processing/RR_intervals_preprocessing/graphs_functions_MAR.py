from tool_box_MAR import *
from RR_intervals_preprocessing_MAR import *
#------------------------------------------------------------------------------------------------------------
# Function to plot tachogram comparison before and after filtering:

def add_tachogram_traces(fig, rr_before, rr_after, color_after, start_row):
    """
    פונקציית עזר שמחשבת זמנים ומוסיפה שתי שורות של טכוגרמה (לפני ואחרי)
    לתוך אובייקט ה-Figure הקיים.
    """
    # חישוב ציר הזמן
    time_before = np.cumsum(rr_before) / 1000
    time_after = np.cumsum(rr_after) / 1000
    
    # ציור הטכוגרמה של "לפני" (בשורה start_row)
    fig.add_trace(
        go.Scatter(x=time_before, y=rr_before, mode='lines+markers', 
                   name='Tacho Before', line=dict(color='gray', width=1), marker=dict(size=3)), 
        row=start_row, col=1
    )
    
    # ציור הטכוגרמה של "אחרי" (בשורה start_row + 1)
    fig.add_trace(
        go.Scatter(x=time_after, y=rr_after, mode='lines+markers', 
                   name='Tacho After', line=dict(color=color_after, width=1), marker=dict(size=4)), 
        row=start_row + 1, col=1
    )

    # הגדרת שמות הצירים לטכוגרמות האלו
    fig.update_yaxes(title_text="RR Duration (ms)", row=start_row, col=1)
    fig.update_yaxes(title_text="RR Duration (ms)", row=start_row + 1, col=1)
    fig.update_xaxes(title_text="Time from start (sec)", row=start_row + 1, col=1)





#------------------------------------------------------------------------------------------------------------

# Function to plot histogram comparison before and after filtering:



def plot_all_filters_comparison(file_path, params, output_folder):
    rr_before = np.load(file_path)
    
    params["apply_rbf"] = True
    params["apply_maf"] = True
    params["apply_qf"] = True
    
    rr_after_rbf = RBF(rr_before, params)
    rr_after_maf = MAF(rr_before, params)
    rr_after_qf = QF(rr_before, params)
    
    # שמירת מרווחי RBF לקובץ NPY
    os.makedirs(output_folder, exist_ok=True)
    file_name = os.path.basename(file_path)
    npy_save_name = file_name.replace('.npy', '_RBF_filtered.npy')
    npy_save_path = os.path.join(output_folder, npy_save_name)
    np.save(npy_save_path, rr_after_rbf)
    
    # חישוב גבולות X להיסטוגרמות
    rbf_min = min(rr_after_rbf) * 0.9 if rr_after_rbf else 0
    rbf_max = max(rr_after_rbf) * 1.1 if rr_after_rbf else 1
    maf_min = min(rr_after_maf) * 0.9 if rr_after_maf else 0
    maf_max = max(rr_after_maf) * 1.1 if rr_after_maf else 1
    qf_min = min(rr_after_qf) * 0.9 if rr_after_qf else 0
    qf_max = max(rr_after_qf) * 1.1 if rr_after_qf else 1
    
    rbf_params = f"(min_HR: {params.get('min_HR')}, max_HR: {params.get('max_HR')})"
    maf_params = f"(window: {params.get('window_size_maf')}, limit: {params.get('limit_value_maf')}%)"
    qf_params = f"(limit: {params.get('limit_value_qf')}%)"

    fig = make_subplots(
        rows=12, cols=1, 
        subplot_titles=(
            f"Histogram BEFORE RBF (Total: {len(rr_before)})", 
            f"Histogram AFTER RBF (Total: {len(rr_after_rbf)}) | {rbf_params}",
            f"Tachogram BEFORE RBF", f"Tachogram AFTER RBF",
            
            f"Histogram BEFORE MAF (Total: {len(rr_before)})", 
            f"Histogram AFTER MAF (Total: {len(rr_after_maf)}) | {maf_params}",
            f"Tachogram BEFORE MAF", f"Tachogram AFTER MAF",
            
            f"Histogram BEFORE QF (Total: {len(rr_before)})", 
            f"Histogram AFTER QF (Total: {len(rr_after_qf)}) | {qf_params}",
            f"Tachogram BEFORE QF", f"Tachogram AFTER QF"
        ),
        vertical_spacing=0.04 
    )

    # --- הוספת ההיסטוגרמות ---
    # RBF
    fig.add_trace(go.Histogram(x=rr_before, name="Hist Before RBF", marker_color='gray', opacity=0.7, xbins=dict(size=5)), row=1, col=1)
    fig.add_trace(go.Histogram(x=rr_after_rbf, name="Hist After RBF", marker_color='blue', opacity=0.7, xbins=dict(size=5)), row=2, col=1)
    # MAF
    fig.add_trace(go.Histogram(x=rr_before, name="Hist Before MAF", marker_color='gray', opacity=0.7, xbins=dict(size=5)), row=5, col=1)
    fig.add_trace(go.Histogram(x=rr_after_maf, name="Hist After MAF", marker_color='green', opacity=0.7, xbins=dict(size=5)), row=6, col=1)
    # QF
    fig.add_trace(go.Histogram(x=rr_before, name="Hist Before QF", marker_color='gray', opacity=0.7, xbins=dict(size=5)), row=9, col=1)
    fig.add_trace(go.Histogram(x=rr_after_qf, name="Hist After QF", marker_color='red', opacity=0.7, xbins=dict(size=5)), row=10, col=1)

    # --- קריאה לפונקציית העזר של הטכוגרמות ---
    # שולחים את fig כדי שהפונקציה תוסיף אליו את הגרפים בשורות המתאימות
    add_tachogram_traces(fig, rr_before, rr_after_rbf, 'blue', start_row=3)
    add_tachogram_traces(fig, rr_before, rr_after_maf, 'green', start_row=7)
    add_tachogram_traces(fig, rr_before, rr_after_qf, 'red', start_row=11)

    # --- עדכון צירי X להיסטוגרמות ---
    fig.update_xaxes(range=[rbf_min, rbf_max], row=2, col=1)
    fig.update_xaxes(range=[maf_min, maf_max], row=6, col=1)
    fig.update_xaxes(range=[qf_min, qf_max], row=10, col=1)
    
    # --- עיצוב ושמירה ---
    fig.update_layout(
        height=3200, 
        bargap=0.05, 
        title_text=f"Complete RR Analysis: {file_name}", 
        template="plotly_white",
        showlegend=False
    )

    # קווי הפרדה עבים
    fig.add_shape(type="line", x0=0, x1=1, y0=0.666, y1=0.666, line=dict(color="LightGray", width=3, dash="dash"), xref="paper", yref="paper")
    fig.add_shape(type="line", x0=0, x1=1, y0=0.333, y1=0.333, line=dict(color="LightGray", width=3, dash="dash"), xref="paper", yref="paper")

    html_save_path = os.path.join(output_folder, file_name.replace('.npy', '_analysis.html'))
    fig.write_html(html_save_path)







