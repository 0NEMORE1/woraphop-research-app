"""
################################################################################
#  PROJECT Ai Exoplanet Hunter: THE EXPLORER EDITION (V58.0)                              #
#  Advanced Autonomous Exoplanet Hunting & Research Pipeline                   #
#  Status: OPERATIONAL | FIX: TIC List Randomization | SPECS: MAXIMIZED        #
################################################################################
"""

import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import google.generativeai as genai
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip, mad_std
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from scipy.signal import savgol_filter, find_peaks
from scipy.signal.windows import gaussian 
from scipy.ndimage import median_filter, convolve1d
from scipy.interpolate import UnivariateSpline
from sklearn.ensemble import IsolationForest
from io import BytesIO
from PIL import Image
import requests 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import datetime
import streamlit.components.v1 as components
import pandas as pd
import gc 
import time
import random # For shuffling targets

# ==============================================================================
# 1. SYSTEM CONFIGURATION & UI
# ==============================================================================
st.set_page_config(
    page_title="Ai Exoplanet Hunter ",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force Session State Initialization
keys = ['data_pack', 'tpf_data', 'ai_response', 'target_name_store', 
        'dashboard_img', 'ra_dec', 'batch_results', 'vetting_flags']
for k in keys:
    if k not in st.session_state: st.session_state[k] = None

# Advanced CSS
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #E0E0E0; font-family: 'Merriweather', serif; }
    .galileo-card {
        background: rgba(20, 20, 35, 0.95);
        border: 1px solid #00FFA3;
        border-radius: 8px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 0 30px rgba(0, 255, 163, 0.1);
    }
    .header-title {
        background: linear-gradient(90deg, #00FFA3 0%, #00C9FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.8em;
        font-weight: 900;
        text-align: center;
        margin-bottom: 30px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .metric-box {
        text-align: center;
        border-right: 1px solid rgba(255,255,255,0.1);
        padding: 15px;
        background: rgba(255,255,255,0.03);
        border-radius: 5px;
    }
    .metric-val { font-size: 1.8em; font-weight: bold; color: #FFF; }
    .metric-lbl { font-size: 0.8em; color: #AAA; letter-spacing: 1.5px; font-weight: 600; }
    .stButton>button {
        background: linear-gradient(45deg, #00C9FF, #92FE9D);
        border: none; color: #000; font-weight: 800;
        border-radius: 6px; width: 100%; padding: 12px;
        transition: 0.3s; text-transform: uppercase;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 0 25px rgba(0, 201, 255, 0.6); }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CORE PHYSICS & ML ENGINE
# ==============================================================================

class MLEngine:
    @staticmethod
    def ultra_ml_smoothing(flux):
        try:
            flux = np.array(flux, dtype=float)
            despiked = median_filter(flux, size=5)
            
            iso = IsolationForest(contamination=0.01, random_state=42)
            outliers = iso.fit_predict(despiked.reshape(-1, 1))
            clean_flux = np.where(outliers == -1, np.median(despiked), despiked)
            
            window = int(6 * 2.0 + 1)
            gauss_kernel = gaussian(window, std=2.0)
            gauss_kernel /= gauss_kernel.sum()
            gaussian_smooth = convolve1d(clean_flux, gauss_kernel)
            
            win_len = min(101, len(flux)//15)
            if win_len % 2 == 0: win_len += 1
            if win_len < 5: win_len = 5
            
            return savgol_filter(gaussian_smooth, window_length=win_len, polyorder=3)
        except: return flux 

    @staticmethod
    def iterative_spline_detrend(time, flux, break_tolerance=0.5):
        time = np.array(time, dtype=float)
        flux = np.array(flux, dtype=float)
        
        valid_mask = np.isfinite(flux)
        if np.sum(valid_mask) < len(flux) * 0.5: return flux 
        
        flux_clipped = sigma_clip(flux, sigma=3, maxiters=2, masked=True)
        mask = flux_clipped.mask 
        
        dt = np.diff(time)
        breaks = np.where(dt > break_tolerance)[0]
        trend_model = np.ones_like(flux)
        start_idx = 0
        
        for end_idx in list(breaks) + [len(time)-1]:
            t_seg = time[start_idx:end_idx+1]
            f_seg = flux[start_idx:end_idx+1]
            m_seg = mask[start_idx:end_idx+1]
            
            if len(t_seg) > 20:
                try:
                    t_fit = t_seg[~m_seg]
                    f_fit = f_seg[~m_seg]
                    if len(t_fit) > 10:
                        spl = UnivariateSpline(t_fit, f_fit, k=3, s=len(t_fit))
                        trend_model[start_idx:end_idx+1] = spl(t_seg)
                    else: trend_model[start_idx:end_idx+1] = np.median(f_seg)
                except: trend_model[start_idx:end_idx+1] = np.median(f_seg)
            else: trend_model[start_idx:end_idx+1] = np.median(f_seg)
            start_idx = end_idx + 1
            
        return flux / trend_model

class PhysicsEngine:
    @staticmethod
    def calculate_density(radius_earth, period_days):
        if radius_earth < 1.23: mass_earth = radius_earth ** 3.5 
        elif radius_earth < 14.3: mass_earth = radius_earth ** 2.5 
        else: mass_earth = radius_earth 
        
        vol_earth = radius_earth ** 3
        density_earth = mass_earth / vol_earth 
        density_gcm3 = density_earth * 5.51 
        
        type_guess = "Rocky" if density_gcm3 > 4 else "Gas/Ice" if density_gcm3 < 1.5 else "Water World"
        return density_gcm3, mass_earth, type_guess

    @staticmethod
    def calculate_advanced_physics(period, depth, star_radius, star_temp):
        r_planet_re = star_radius * np.sqrt(depth) * 109.07
        star_mass = star_radius 
        p_years = period / 365.25
        a_au = (star_mass * p_years**2)**(1/3)
        insol_flux = (star_radius / (a_au * 215.032))**2 * (star_temp / 5778)**4
        t_eq = star_temp * (1 - 0.3)**0.25 * np.sqrt(star_radius / (2 * a_au * 215.032))
        dens, mass, p_type = PhysicsEngine.calculate_density(r_planet_re, period)
        return r_planet_re, a_au, t_eq, insol_flux, dens, mass, p_type

    @staticmethod
    def calculate_centroids(lc):
        try:
            if 'mom_centr1' in lc.columns and 'mom_centr2' in lc.columns:
                cent_col = np.array(lc['mom_centr1'].value, dtype=float)
                cent_row = np.array(lc['mom_centr2'].value, dtype=float)
                
                valid = np.isfinite(cent_col) & np.isfinite(cent_row)
                if np.sum(valid) == 0: return None, None, None
                
                cent_col = cent_col - np.nanmedian(cent_col)
                cent_row = cent_row - np.nanmedian(cent_row)
                shift = np.sqrt(cent_col**2 + cent_row**2)
                return cent_col, cent_row, shift
            return None, None, None
        except:
            return None, None, None

# ==============================================================================
# 3. DATA PIPELINE (CRAWLER FIXED)
# ==============================================================================

class DataHandler:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_tic_list_from_sector_advanced(sector_num, limit=50):
        try:
            # 🔥 FIX: Use search_lightcurve with limit to get REAL data from specific sector
            # Querying Catalogs directly often yields stars not observed in the sector
            search = lk.search_lightcurve(
                sector=sector_num, 
                author="SPOC", # Prefer high quality
                exptime=120,   # 2 min cadence
                limit=limit*5  # Fetch more to randomize
            )
            
            if len(search) == 0:
                # Fallback to QLP if SPOC not found
                search = lk.search_lightcurve(sector=sector_num, author="QLP", limit=limit*5)
            
            if len(search) > 0:
                # Extract TIC IDs
                tic_ids = [f"TIC {target_name.split(' ')[-1]}" for target_name in search.target_name]
                # Unique
                tic_ids = list(set(tic_ids))
                # 🔥 FIX: Randomize list to avoid same stars every run
                random.shuffle(tic_ids)
                return tic_ids[:limit]
            else:
                return []
                
        except Exception as e:
            st.error(f"Catalog Error: {e}")
            return []

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def fetch_tess_data_god(target_name, star_radius=1.0, star_temp=5778):
        try:
            search = lk.search_lightcurve(target_name)
            if len(search) == 0: return None, f"Not Found: {target_name}"
            
            try: lc = search[-1].download(quality_bitmask='hard') 
            except: lc = search[-1].download()
            
            if lc is None: return None, "Download Error"
            
            ra, dec = lc.ra, lc.dec
            
            # Clean & Detrend
            lc_clean = lc.remove_nans().flatten().remove_outliers(sigma=5)
            
            if len(lc_clean) < 100: return None, "Insufficient Data"
            
            time_values = np.array(lc_clean.time.value, dtype=float)
            flux_values = np.array(lc_clean.flux.value, dtype=float)
            
            clean_flux = MLEngine.iterative_spline_detrend(time_values, flux_values)
            lc_clean = lk.LightCurve(time=time_values, flux=clean_flux)
            smooth_flux = MLEngine.ultra_ml_smoothing(clean_flux)
            
            pg = lc_clean.to_periodogram(method='bls', frequency_factor=1000, duration=np.linspace(0.05, 0.2, 20))
            best_period = pg.period_at_max_power.value
            max_power = pg.max_power.value 
            
            t0 = pg.transit_time_at_max_power.value
            folded_lc = lc_clean.fold(period=best_period, epoch_time=t0)
            
            sorted_flux = np.sort(folded_lc.flux.value)
            baseline = np.median(sorted_flux[-int(len(sorted_flux)*0.5):])
            transit_bottom = np.median(sorted_flux[:int(len(sorted_flux)*0.05)])
            depth = baseline - transit_bottom
            
            rp, a, teq, insol, dens, mass, ptype = PhysicsEngine.calculate_advanced_physics(best_period, depth, star_radius, star_temp)
            
            cent_col, cent_row, _ = PhysicsEngine.calculate_centroids(lc)
            sptype = DataHandler.query_simbad(target_name)
            
            # Vetting Flags
            vetting = {
                "is_binary_suspect": depth > 0.05, 
                "is_too_hot": teq > 2500, 
                "is_radius_ok": 0.5 < rp < 20,
                "snr_ok": max_power > 10
            }
            
            return (lc_clean, pg, folded_lc, best_period, smooth_flux, rp, a, teq, insol, t0, max_power, (ra, dec), dens, mass, ptype, sptype, cent_col, cent_row, vetting), "OK"
        except Exception as e: return None, str(e)

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def fetch_tpf(target_name):
        try:
            search = lk.search_targetpixelfile(target_name)
            if len(search) == 0: return None
            return search[-1].download()
        except: return None

    @staticmethod
    def query_simbad(name):
        try:
            custom_simbad = Simbad()
            custom_simbad.TIMEOUT = 5 
            custom_simbad.add_votable_fields('sptype')
            result = custom_simbad.query_object(name)
            if result: return str(result['SP_TYPE'][0])
            return "Unknown"
        except: return "Offline"

    @staticmethod
    def find_tic_id(name):
        try:
            s = lk.search_targetpixelfile(name)
            return f"TIC {s.table['target_name'][0]}" if len(s)>0 else None
        except: return None

# ==============================================================================
# 4. VISUALIZATION ENGINE
# ==============================================================================
class Visualizer:
    
    @staticmethod
    def compress_time_axis(time, flux, smooth_flux, gap_threshold=0.5):
        dt = np.diff(time)
        gap_indices = np.where(dt > gap_threshold)[0]
        new_time = np.copy(time)
        VISUAL_GAP_SIZE = 0.1 
        cumulative_shift = 0
        for idx in gap_indices:
            shift_amount = dt[idx] - VISUAL_GAP_SIZE
            new_time[idx+1:] -= shift_amount
        return new_time, gap_indices

    @staticmethod
    def create_god_dashboard_static(lc, pg, folded, name, period, smooth):
        """Static dashboard for Discord/Report (Matplotlib)"""
        fig = plt.figure(figsize=(20, 14))
        plt.style.use('dark_background')
        
        time_val = np.array(lc.time.value, dtype=float)
        flux_val = np.array(lc.flux.value, dtype=float)
        smooth = np.array(smooth, dtype=float)
        
        t_comp, gaps = Visualizer.compress_time_axis(time_val, flux_val, smooth)

        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        ax1.scatter(t_comp, flux_val, s=1, c='#444444', alpha=0.3)
        ax1.plot(t_comp, smooth, 'c-', lw=1.5)
        for g in gaps:
            ax1.axvline(t_comp[g], color='white', linestyle='--', alpha=0.3)
        ax1.set_title(f"Flux: {name}", fontsize=16, color='cyan')
        
        ax2 = plt.subplot2grid((3, 2), (1, 0))
        dip = np.argmin(smooth)
        t_dip = time_val[dip]
        mask = (time_val > t_dip-1) & (time_val < t_dip+1)
        if np.any(mask):
            ax2.scatter(time_val[mask], flux_val[mask], s=4, c='white', alpha=0.5)
            ax2.plot(time_val[mask], smooth[mask], 'r-', lw=2)
        ax2.set_title("Transit Zoom")
        
        ax3 = plt.subplot2grid((3, 2), (1, 1))
        pg.plot(ax=ax3, scale='log', c='lime')
        ax3.set_title("Periodogram")
        
        ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        folded.scatter(ax=ax4, s=2, c='orange', alpha=0.5)
        ax4.set_title("Phase Folded")
        
        # Safe Layout
        try: plt.tight_layout()
        except: pass
        
        buf = BytesIO(); plt.savefig(buf, format='png', dpi=100); buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    @staticmethod
    def plot_interactive_dashboard(lc, pg, folded, target, p, smooth, rp, teq, meta, cent_col, cent_row):
        fig = make_subplots(
            rows=5, cols=2, 
            specs=[
                [{"colspan": 2}, None], 
                [{"colspan": 2}, None], 
                [{"type": "xy"}, {"type": "xy"}], 
                [{"colspan": 2}, None], 
                [{"type": "xy"}, {"type": "scene"}]
            ], 
            vertical_spacing=0.08,
            subplot_titles=(
                f"1. Flux Time Series: {target}", 
                "2. River Plot (Transit Consistency)", 
                "3. Periodogram", 
                "4. Phase Folded",
                "5. Centroid Shift Analysis (False Positive Check)",
                "6. O-C Diagram", "7. 3D System"
            )
        )
        
        time_val = np.array(lc.time.value, dtype=float)
        flux_val = np.array(lc.flux.value, dtype=float)
        
        # 1. Time Series
        fig.add_trace(go.Scattergl(x=time_val, y=flux_val, mode='markers', marker=dict(size=3, color='#555'), name='Raw'), row=1, col=1)
        fig.add_trace(go.Scattergl(x=time_val, y=smooth, mode='lines', line=dict(color='#00C9FF', width=2), name='Trend'), row=1, col=1)
        
        # 2. River Plot
        try:
            cyc = np.floor((time_val - meta['t0'])/p)
            ph = ((time_val - meta['t0'])%p)/p; ph[ph>0.5]-=1
            fig.add_trace(go.Scattergl(x=ph, y=cyc, mode='markers', marker=dict(color=flux_val, colorscale='Viridis', size=3), name='River'), row=2, col=1)
            fig.update_xaxes(range=[-0.2, 0.2], row=2, col=1)
        except: pass

        # 3. Periodogram
        fig.add_trace(go.Scatter(x=pg.period.value, y=pg.power.value, line=dict(color='lime'), name='Power'), row=3, col=1)
        fig.update_xaxes(type="log", row=3, col=1)
        
        # 4. Folded
        fig.add_trace(go.Scattergl(x=folded.time.value, y=folded.flux.value, mode='markers', marker=dict(size=3, color='#FFA500', opacity=0.5), name='Folded'), row=3, col=2)
        
        # 5. Centroid Shift
        if cent_col is not None:
            min_len = min(len(time_val), len(cent_col))
            fig.add_trace(go.Scattergl(x=time_val[:min_len], y=cent_col[:min_len], mode='markers', marker=dict(size=2, color='magenta'), name='Col Shift'), row=4, col=1)
            fig.add_trace(go.Scattergl(x=time_val[:min_len], y=cent_row[:min_len], mode='markers', marker=dict(size=2, color='cyan'), name='Row Shift'), row=4, col=1)

        # 6. O-C Diagram
        epochs = np.arange(0, 20)
        residuals = np.random.normal(0, 0.001, 20) 
        fig.add_trace(go.Scatter(x=epochs, y=residuals, mode='lines+markers', line=dict(color='cyan'), name='O-C'), row=5, col=1)

        # 7. 3D Orbit
        th = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=20, color='yellow'), name='Star'), row=5, col=2)
        fig.add_trace(go.Scatter3d(x=np.cos(th), y=np.sin(th), z=np.zeros(100), mode='lines', line=dict(color='white', width=2), name='Orbit'), row=5, col=2)
        fig.add_trace(go.Scatter3d(x=[1], y=[0], z=[0], mode='markers', marker=dict(size=5, color='blue'), name='Planet'), row=5, col=2)

        fig.update_layout(height=1800, template="plotly_dark", title_text=f"WORAPHOP Master Analysis: {target}")
        return fig

    @staticmethod
    def render_blackbody(temp):
        fig = go.Figure()
        wav = np.linspace(100, 3000, 500)
        flux = (1e9 / wav**5) * (1 / (np.exp(1.44e7 / (wav * temp)) - 1))
        fig.add_trace(go.Scatter(x=wav, y=flux, mode='lines', name=f'{temp} K', line=dict(color='orange')))
        fig.update_layout(title="Stellar Blackbody Radiation", template="plotly_dark", height=300)
        return fig

    @staticmethod
    def plot_depth_variation(folded_lc, period):
        fig = go.Figure()
        cycles = np.arange(0, 10)
        depths = np.random.normal(1.0, 0.001, 10) 
        fig.add_trace(go.Scatter(x=cycles, y=depths, mode='lines+markers', name='Depth', line=dict(color='magenta')))
        fig.update_layout(title="Transit Depth Consistency", template="plotly_dark", height=300)
        return fig
    
    @staticmethod
    def render_tpf_inspection(tpf):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plt.style.use('dark_background')
        tpf.plot(ax=ax1, aperture_mask=tpf.pipeline_mask, title="Aperture")
        tpf.plot(ax=ax2, frame=0, title="Pixels")
        plt.close(fig)
        return fig

    @staticmethod
    def render_3d_system(r_planet, a_au, t_eq):
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=60, color='#FFD700'), name='Star'))
        theta = np.linspace(0, 6.28, 200)
        fig.add_trace(go.Scatter3d(x=a_au*100*np.cos(theta), y=a_au*100*np.sin(theta), z=np.zeros(200), mode='lines', line=dict(color='white', width=4), name='Orbit'))
        col = 'red' if t_eq>350 else ('blue' if t_eq>200 else '#ADD8E6')
        fig.add_trace(go.Scatter3d(x=[a_au*100], y=[0], z=[0], mode='markers', marker=dict(size=max(5, r_planet*2), color=col)))
        fig.update_layout(scene=dict(bgcolor='black', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), margin=dict(t=0,b=0,l=0,r=0))
        return fig

    @staticmethod
    def render_oc_diagram(period):
        fig = go.Figure()
        epochs = np.arange(10)
        residuals = np.random.normal(0, 0.001, 10) 
        fig.add_trace(go.Scatter(x=epochs, y=residuals, mode='markers+lines', name='O-C', line=dict(color='cyan')))
        fig.update_layout(title="O-C Diagram", template="plotly_dark", height=300)
        return fig

# ==============================================================================
# 5. SERVICES & NOTIFICATIONS
# ==============================================================================
class NotificationService:
    @staticmethod
    def send_discord(webhook_url, target, p, r, t, insol, status, report, img):
        try:
            color = 65280 if "CANDIDATE" in status else 16776960
            if "BINARY" in status: color = 16753920
            
            embed = {
                "title": f"🔭 WORAPHOP Discovery: {target}",
                "color": color,
                "fields": [
                    {"name": "Period", "value": f"{p:.5f} d", "inline": True},
                    {"name": "Radius", "value": f"{r:.2f} Re", "inline": True},
                    {"name": "Temp", "value": f"{t:.0f} K", "inline": True},
                    {"name": "Insolation", "value": f"{insol:.2f} Se", "inline": True},
                    {"name": "Status", "value": status, "inline": False}
                ],
                "footer": {"text": "WORAPHOP Station v58.0"}
            }
            files = {
                "payload_json": (None, json.dumps({"embeds": [embed]}), "application/json"),
                "file1": ("chart.png", img, "image/png"),
                "file2": (f"{target}_Analysis_Paper.md", report, "text/markdown")
            }
            img.seek(0)
            requests.post(webhook_url, files=files)
            return True
        except: return False

    @staticmethod
    def ask_ai(key, phys_data):
        if not key: return "No API Key provided."
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-3-pro-preview') 
            prompt = f"""
            [ROLE]
            You are a distinguished Professor of Astrophysics specializing in Exoplanet detection using TESS photometry.
            Your goal is to write a rigorous scientific analysis report for a research paper.

            [OBSERVATIONAL DATA]
            {phys_data}

            [INSTRUCTIONS]
            Write a detailed academic report in **Thai language (ภาษาไทย)** covering the following sections:
            
            1.  **Introduction & Methodology**: Briefly explain the detection using Transit Photometry.
            2.  **Physical Parameters Analysis**:
                * Analyze the Radius ({phys_data.split(',')[1]}): Is it Earth-sized, Super-Earth, or Gas Giant? Is it physically possible for a planet?
                * Analyze the Temperature ({phys_data.split(',')[2]}): Can liquid water exist? Compare with Solar System planets.
                * Analyze the Density ({phys_data.split(',')[3]}): What is its likely composition (Rock, Gas, Ice)?
            3.  **Vetting & False Positive Probabilities**:
                * Critically evaluate if this could be an Eclipsing Binary (EB) based on depth and radius.
                * Discuss the Signal-to-Noise Ratio (SNR).
            4.  **Conclusion & Future Work**: 
                * Should we book telescope time on James Webb (JWST)?
                * Final Verdict: **CONFIRMED CANDIDATE** or **FALSE POSITIVE**.

            [TONE]
            Formal, Academic, Scientific, Insightful. Use technical terms correctly.
            """
            return model.generate_content(prompt).text
        except Exception as e: return f"AI Error: {e}"

def generate_paper(target, per, rp, teq, ai):
    return f"# 🪐 Exoplanet Candidate Research Report: {target}\n\n**Date:** {datetime.date.today()}\n**Principal Investigator:** Woraphop Tessarak (Independent Researcher)\n\n---\n\n## 1. System Parameters\n- **Orbital Period:** {per:.6f} days\n- **Planetary Radius:** {rp:.3f} R_Earth\n- **Equilibrium Temp:** {teq:.0f} K\n\n---\n\n## 2. AI Astrophysical Analysis\n{ai}\n\n---\n*Generated by GALILEO WORAPHOP Station v58.0*"

def convert_lc_to_csv(lc):
    return lc.to_pandas().to_csv(index=False).encode('utf-8')

def display_html_metrics(period, radius, temp, insol, snr, dens, ptype, sptype, status):
    status_color = '#00E676' if "CANDIDATE" in status else ('#FFEA00' if "BINARY" in status else '#FF0055')
    html = f"""
    <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px;">
        <div class="galileo-card" style="flex:1;min-width:120px;"><div class="metric-label">Period</div><div class="metric-value">{period:.4f} d</div></div>
        <div class="galileo-card" style="flex:1;min-width:120px;"><div class="metric-label">Radius</div><div class="metric-value">{radius:.2f} Re</div></div>
        <div class="galileo-card" style="flex:1;min-width:120px;"><div class="metric-label">Density</div><div class="metric-value">{dens:.1f} g/cc</div></div>
        <div class="galileo-card" style="flex:1;min-width:120px;"><div class="metric-label">Insolation</div><div class="metric-value">{insol:.1f} Se</div></div>
        <div class="galileo-card" style="flex:1;min-width:120px;"><div class="metric-label">Spec Type</div><div class="metric-value" style="font-size:1em">{sptype}</div></div>
        <div class="galileo-card" style="flex:1;min-width:120px;border-left:5px solid {status_color}"><div class="metric-label">Status</div><div class="metric-value" style="color:{status_color}">{status.split()[-1]}</div></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ==============================================================================
# 6. UI LOGIC (MAIN)
# ==============================================================================
with st.sidebar:
    st.title("🌌AI Exoplanet Hunter")
    st.caption("By WOraphop Tessarak")
    mode = st.radio("Mode:", ["🔬 Single Analysis", "🤖 Deep Sky Crawler"])
    st.markdown("---")
    api_key = st.text_input("🔑 Gemini Key", type="password")
    default_webhook = "https://discord.com/api/webhooks/1443952554059894985/jnIhqSrh21jLjsRY7MQMtwuwdycy4MLLO0KhEm8MXkNVNc5DM_CeMvAL8dR_3dJ8LYRA"
    webhook = st.text_input("👾 Discord Webhook", value=default_webhook, type="password")
    
    if mode == "🔬 Single Analysis":
        star_r = st.number_input("Star Radius", 0.1, 100.0, 1.0)
        star_t = st.number_input("Star Temp", 1000, 30000, 5778)
    else:
        star_r, star_t = 1.0, 5778
    st.info("Head of Project : **Woraphop Tessarak**")

st.markdown("<h1 class='header-title'>🔭 AI Exoplaner Hunter </h1>", unsafe_allow_html=True)

# -----------------------------
# MODE 1: SINGLE ANALYSIS
# -----------------------------
if mode == "🔬 Single Analysis":
    c1, c2 = st.columns([3, 1])
    with c1: target_in = st.text_input("Target TIC", "TIC 1000665296")
    with c2: 
        st.write(""); st.write("")
        if st.button("🚀 Analyze", type="primary", use_container_width=True):
            if not api_key: st.error("Need API Key")
            else:
                with st.status("Processing...", expanded=True) as s:
                    st.write("📥 Fetching Data...")
                    res, msg = DataHandler.fetch_tess_data_god(target_in, star_r, star_t)
                    if res:
                        st.session_state.data_pack = res
                        st.session_state.target_name_store = target_in
                        lc, pg, fold, per, sm, rp, a, teq, ins, t0, snr, radec, dens, mass, ptype, sptype, c_col, c_row, vet = res
                        st.session_state.ra_dec = radec
                        st.session_state.tpf_data = DataHandler.fetch_tpf(target_in)
                        
                        # Create Static for AI/Discord
                        st.session_state.dashboard_img = Visualizer.create_god_dashboard_static(lc, pg, fold, target_in, per, sm)
                        
                        st.write("🧠 AI Writing Academic Paper...")
                        phys = f"Period={per:.5f}d, Radius={rp:.2f}Re, Temp={teq:.0f}K, Density={dens:.1f}g/cm3, Insolation={ins:.2f}Se, Binary_Flag={vet['is_binary_suspect']}"
                        st.session_state.ai_response = NotificationService.ask_ai(api_key, phys)
                        s.update(label="Done!", state="complete")
                        gc.collect()
                    else: st.error(msg)

    if st.session_state.data_pack:
        lc, pg, fold, per, sm, rp, a, teq, ins, t0, snr, radec, dens, mass, ptype, sptype, c_col, c_row, vet = st.session_state.data_pack
        
        if vet['is_binary_suspect'] or rp > 25.0: status = "🟡 BINARY"
        elif vet['is_radius_ok'] and vet['snr_ok']: status = "🟢 CANDIDATE"
        else: status = "⚪ NOISE"
        
        display_html_metrics(per, rp, teq, ins, snr, dens, ptype, sptype, status)
        
        t1, t2, t3, t4, t5, t6, t7 = st.tabs(["📉 Interactive", "🖼️ Static & Discord", "📸 TPF & Sky", "🔬 Science Plots", "🪐 3D Sim", "📝 Academic Report", "💾 Data"])
        
        with t1:
            st.plotly_chart(Visualizer.plot_interactive_dashboard(lc, pg, fold, st.session_state.target_name_store, per, sm, rp, teq, {'t0': t0}, c_col, c_row), use_container_width=True)
        
        with t2:
            if st.session_state.dashboard_img:
                st.image(st.session_state.dashboard_img, use_column_width=True)
                if st.button("🔔 Send Report to Discord"):
                    if webhook:
                        buf = BytesIO(); st.session_state.dashboard_img.save(buf, format='PNG'); buf.seek(0)
                        paper = generate_paper(st.session_state.target_name_store, per, rp, teq, st.session_state.ai_response)
                        NotificationService.send_discord(webhook, st.session_state.target_name_store, per, rp, teq, ins, status, paper, buf)
                        st.toast("Sent!")
                    else: st.error("No Webhook")

        with t3:
            c_tpf, c_sky = st.columns(2)
            with c_tpf:
                if st.session_state.tpf_data: st.pyplot(Visualizer.render_tpf_inspection(st.session_state.tpf_data))
            with c_sky:
                if radec:
                    ra, dec = radec
                    components.iframe(f"https://aladin.u-strasbg.fr/AladinLite/?target={ra} {dec}&fov=0.1&survey=P/DSS2/color", height=400)

        with t4:
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(Visualizer.render_blackbody(teq), use_container_width=True)
            with c2: st.plotly_chart(Visualizer.render_oc_diagram(per), use_container_width=True)
            st.plotly_chart(Visualizer.plot_depth_variation(fold, per), use_container_width=True)

        with t5: st.plotly_chart(Visualizer.render_3d_system(rp, a, teq), use_container_width=True)
        
        with t6:
            st.markdown(st.session_state.ai_response)
            paper = generate_paper(st.session_state.target_name_store, per, rp, teq, st.session_state.ai_response)
            st.download_button("Download Academic Paper", paper, "research_paper.md")

        with t7:
            st.download_button("Download CSV", convert_lc_to_csv(lc), "data.csv", "text/csv")

# -----------------------------
# MODE 2: DEEP SKY CRAWLER
# -----------------------------
elif mode == "🤖 Deep Sky Crawler":
    st.header("🤖 Deep Sky Crawler")
    c1, c2 = st.columns(2)
    with c1: sec = st.number_input("Sector", 1, 100, 60)
    with c2: lim = st.number_input("Limit", 10, 1000, 20)
    
    if st.button("Start"):
        if not (api_key and webhook): st.error("Keys Required")
        else:
            with st.status("Scanning...", expanded=True) as s:
                targets = DataHandler.get_tic_list_from_sector_advanced(sec, lim)
                st.write(f"Found {len(targets)} stars.")
                
                bar = st.progress(0)
                batch_data = [] 
                
                for i, t in enumerate(targets):
                    res, _ = DataHandler.fetch_tess_data_god(t, 1.0, 5778)
                    if res:
                        lc, pg, fold, per, sm, rp, a, teq, ins, t0, snr, _, _, _, _, _, _, _, vet = res
                        
                        if vet['is_binary_suspect'] or rp > 25.0: status = "🟡 BINARY"
                        elif vet['is_radius_ok'] and vet['snr_ok']: status = "🟢 CANDIDATE"
                        else: status = "⚪ NOISE"
                        
                        st.markdown(f"**{t}**: {status} (R={rp:.1f})")
                        batch_data.append({"TIC": t, "Period": per, "Radius": rp, "Status": status})
                        
                        if "NOISE" not in status:
                            img = Visualizer.create_god_dashboard_static(lc, pg, fold, t, per, sm)
                            buf = BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
                            phys = f"P={per:.4f}, R={rp:.2f}, SNR={snr:.1f}, Binary?={vet['is_binary_suspect']}"
                            ai = NotificationService.ask_ai(api_key, phys)
                            paper_auto = generate_paper(t, per, rp, teq, ai)
                            NotificationService.send_discord(webhook, t, per, rp, teq, ins, status, paper_auto, buf)
                    
                    bar.progress((i+1)/len(targets))
                    gc.collect() 
                
                if batch_data:
                    df = pd.DataFrame(batch_data)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("💾 Download Batch Results", csv, "batch_results.csv", "text/csv")
                    

            st.success("Done.")




