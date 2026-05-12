import streamlit as st
from main import ECG_Pipeline, Advanced_ECG_Pipeline

# basic page setup
st.set_page_config(page_title="ECG DSP Viewer", layout="wide")
st.title("ECG DSP Pipeline Viewer")

# sidebar controls
st.sidebar.header("Test Parameters")
selected_patient = st.sidebar.selectbox("MIT-BIH Record", [100, 104, 208, 217, 232])
inject_noise = st.sidebar.checkbox("Inject 60Hz Noise (Adv Tab)", value=True)
fir_window = st.sidebar.slider("FIR Window Size (Taps, Adv Tab)", min_value=5, max_value=55, value=15, step=2)

# setup tabs
tab1, tab2 = st.tabs(["Clinical View", "DSP Math (Advanced)"])

with tab1:
    st.subheader("Standard Pipeline (Butterworth + HRV)")
    
    if st.button("Run Analysis", key="run_base"):
        with st.spinner("Processing..."):
            
            # init and run base pipeline
            base = ECG_Pipeline(selected_patient, run_batch=False)
            base.load_data()
            base.apply_bandpass()
            base.find_peaks()
            base.calculate_bpm()
            base.calculate_hrv()

            # basic metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("BPM", round(base.bpm, 1))
            col2.metric("SDNN (ms)", round(base.sdnn, 2))
            col3.metric("Anomalies Filtered", base.pvc_count)

            # fetch and render plot
            st.pyplot(base.plot_results(return_fig=True))

with tab2:
    st.subheader("Frequency Domain & Custom FIR")
    st.write("Isolating phase delay and 60Hz powerline interference.")
    
    with st.spinner("Crunching FFT..."):
        
        # init and run advanced subclass
        adv = Advanced_ECG_Pipeline(selected_patient, run_batch=False)
        adv.load_data()
        adv.custom_fir_filter(window_size=fir_window) # Your slider variable!
        adv.analyze_frequency(inject_noise=inject_noise)

        # engineering metrics
        col1, col2 = st.columns(2)
        col1.metric("FIR Window (Taps)", fir_window)
        col2.metric("60Hz Noise", "Injected" if inject_noise else "None")

        # fetch and render plot
        st.pyplot(adv.plot_results(return_fig=True))