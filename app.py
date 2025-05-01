import streamlit as st
import platform
st.write("Python version:", platform.python_version())
import numpy as np
import tensorflow as tf
import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model
import os
import librosa



# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Load the model
try:
    model = load_model('best_model.h5', compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# Pre-computed mean and std of training set
clinical_means = np.array([
    40.85067873, 161.7498188, 57.69040724, 47.03167421,
    0.180580762, 0.162895928, 0.015384615, 0.00361991,
    0.133936652, 86.17013575, 36.72171946, 0.56561086,
    0.190045249, 0.449773756, 0.43800905, 0.5
])
clinical_stds = np.array([
    15.28771908, 8.792234508, 13.67719965, 54.9715837,
    0.384670444, 0.369270693, 0.123076923, 0.060056688,
    0.340584241, 16.73363358, 0.541607072, 0.495676523,
    0.392336657, 0.497470928, 0.49614224, 0.5
])

# ✅ Modified: Preprocess audio with resampling + loudest segment extraction
def preprocess_audio(audio_data, orig_sr):
    target_sr = 44100
    target_length = 22050  # 0.5 sec at 44.1kHz

    if orig_sr != target_sr:
        audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)

    audio_data = (audio_data - np.mean(audio_data)) / (np.std(audio_data) + 1e-10)

    if len(audio_data) > target_length:
        max_energy = 0
        best_start = 0
        hop = int(target_sr * 0.1)
        for start in range(0, len(audio_data) - target_length + 1, hop):
            window = audio_data[start:start + target_length]
            energy = np.sum(window ** 2)
            if energy > max_energy:
                max_energy = energy
                best_start = start
        audio_data = audio_data[best_start:best_start + target_length]
    else:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')

    return audio_data.reshape(1, -1, 1).astype(np.float32)

# Preprocess clinical data
def preprocess_clinical_data(raw_features):
    standardized = (raw_features - clinical_means) / (clinical_stds + 1e-8)
    return standardized.reshape(1, -1)

def main():
    st.title('Prescreening Tuberculosis (TB) Based on Cough Sound Analysis and Clinical Information Using Machine Learning')
    tabs = ["Record Cough Sound", "Upload Cough Sound"]
    choice = st.sidebar.selectbox("Choose Option", tabs)

    audio_ready = False
    audio_file = None

    if choice == "Record Cough Sound":
        st.write("**Record Cough Sound**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Recording"):
                try:
                    duration = 5
                    sample_rate = 22050
                    st.info("Recording... Please cough into your microphone.")
                    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
                    sd.wait()
                    write("audio_file.wav", sample_rate, audio_data)
                    st.session_state['audio_recorded'] = True
                    st.success("Recording saved successfully!")
                except Exception as e:
                    st.error(f"Error during recording: {str(e)}")
        with col2:
            if st.button("Play Recording"):
                if os.path.exists("audio_file.wav"):
                    audio_file = "audio_file.wav"
                    audio_ready = True
                    st.audio(audio_file)
                else:
                    st.error("No recording found. Please record first.")
    else:
        st.write("**Upload Cough Sound**")
        uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
        if uploaded_file is not None:
            try:
                with open("audio_file.wav", "wb") as f:
                    f.write(uploaded_file.getvalue())
                audio_file = "audio_file.wav"
                audio_ready = True
                st.success("File uploaded successfully!")
                st.audio(uploaded_file)
            except Exception as e:
                st.error(f"Error during file upload: {str(e)}")

    st.write('**Clinical Information**')
    age = st.number_input('Age', 1, 100, 30)
    height = st.number_input('Height (cm)', 100, 300, 170)
    weight = st.number_input('Weight (kg)', 20, 200, 70)
    reported_cough_dur = st.number_input('Cough Duration (days)', 1, 100, 10)
    tb_prior = st.radio('Previous TB Diagnosis?', ('No', 'Yes'))
    tb_prior_pul = st.radio('Previous Pulmonary TB?', ('No', 'Yes'))
    tb_prior_extrapul = st.radio('Previous Extrapulmonary TB?', ('No', 'Yes'))
    tb_prior_unknown = st.radio('Previous TB Unknown Type?', ('No', 'Yes'))
    hemoptysis = st.radio('Hemoptysis (Coughing blood)?', ('No', 'Yes'))
    heart_rate = st.number_input('Heart Rate (bpm)', 50, 200, 80)
    temperature = st.number_input('Temperature (°C)', 35.0, 40.0, 37.0)
    weight_loss = st.radio('Weight Loss?', ('No', 'Yes'))
    smoke_lweek = st.radio('Smoked in Last Week?', ('No', 'Yes'))
    fever = st.radio('Fever?', ('No', 'Yes'))
    night_sweats = st.radio('Night Sweats?', ('No', 'Yes'))
    sex = st.radio('Sex', ('Male', 'Female'))

    if st.button('Predict'):
        if not os.path.exists("audio_file.wav"):
            st.error("Please record or upload an audio file first!")
            return
        try:
            audio_data, rate = sf.read("audio_file.wav")
            audio_input = preprocess_audio(audio_data, orig_sr=rate)

            raw_clinical_features = np.array([
                age, height, weight, reported_cough_dur,
                int(tb_prior == 'Yes'), int(tb_prior_pul == 'Yes'),
                int(tb_prior_extrapul == 'Yes'), int(tb_prior_unknown == 'Yes'),
                int(hemoptysis == 'Yes'), heart_rate, temperature,
                int(weight_loss == 'Yes'), int(smoke_lweek == 'Yes'),
                int(fever == 'Yes'), int(night_sweats == 'Yes'),
                int(sex == 'Male')
            ], dtype=np.float32)

            clinical_input = preprocess_clinical_data(raw_clinical_features)
            prediction = model.predict([audio_input, clinical_input])
            probability = prediction[0][0]

            st.write(f"**TB Probability:** {probability:.2%}")
            if probability >= 0.5:
                st.warning("High probability of tuberculosis (TB).Further clinical evaluation is strongly recommended.")
            else:
                st.success("Low probability of tuberculosis (TB).Additional testing may be considered if symptoms persist or other risk factors are present.")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
