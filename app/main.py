"""Streamlit Webapp to handle offline transcriptions"""
from pathlib import Path
from tempfile import NamedTemporaryFile
import codecs

from streamlit.runtime.scriptrunner import get_script_run_ctx
import streamlit as st

from converters import vtt2docx, vtt2pdf, vtt2txt
from transcribers import whisperx

# set the details of the page
st.set_page_config(
    page_title="Offline Single Transcription",
    layout="centered",
    initial_sidebar_state="auto"
)

# create transcript session state
if 'transcript' not in st.session_state:
    st.session_state['transcript'] = None

if 'export' not in st.session_state:
    st.session_state['export'] = 'vtt'
    st.session_state['disabled'] = True

# create dummy transcript file if doesn't exist.
if 'transcript_file' not in st.session_state:
    Path('./app/audio').mkdir(parents=True, exist_ok=True)
    transcript_file_path = Path('./app/audio/transcript.txt')
    with open(transcript_file_path, 'w+', encoding='utf-8') as tt:
        tt.write(" ")

    st.session_state['transcript_file'] = transcript_file_path
    st.session_state['transcript_output'] = st.session_state['transcript_file']

# get IP of remote client
def get_remote_ip() -> str:
    """Returns the remote ip for this each session."""
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        session_info = st.runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception as e:
        return None

    return session_info.request.remote_ip

# create ip address session state
if 'ip_address' not in st.session_state:
    st.session_state['ip_address'] = get_remote_ip()
    with open('connections.txt', 'a', encoding='utf-8') as cn:
        cn.write(st.session_state['ip_address'] + '\n')

# get the hf token from global secret file
st.session_state['hf_token'] = st.secrets.hf_credentials.hftoken

# transcribe the audio
#@st.cache_resource(show_spinner="Transcribing...")
def transcription(audio_file_name, model):
    """Uses different packages to transcribe audio file and writes the segments"""
    st.session_state['transcript_file'] = whisperx.transcribe(audio_file_name, model, 
                st.session_state['eo'], st.session_state['par'], 1, st.session_state['max_speaker'],
		st.session_state['hf_token'])

    with codecs.open(st.session_state['transcript_file'], encoding='utf-8') as file:
        data = file.read()

    st.session_state['transcript'] = data
    st.session_state['transcript_output'] = st.session_state['transcript_file']

    return True

if __name__ == "__main__":
    # ------------------- Sidebar Information -------------------------
    st.title('Offline Transcription Service')

    # UC banner
    st.sidebar.image('./img/UCWhite.png')
    # Add a description in the sidebar
    st.sidebar.title('About this app')
    st.sidebar.markdown("""This app uses the offline version of the openAI Whisper Automatic Speech
                        Recognition (ASR) package to transcribe uploaded audio or video files.
                        To transcribe an audio or video file, drag and drop the file, or you can use
                        the **Browse Files** button""")

    st.sidebar.subheader("Note")
    st.sidebar.markdown("""This Whisper ASR package has been trained using machine learning, but is
                        secure to use at UC. Your data will not be used to train future versions of
                        the app. All files are automatically deleted after closing the browser window.
                        Please ensure that you download the generated transcript file and save it in a
                        secure location. You should also save the original audio or video file in a
                        secure location.
                        """)

    st.sidebar.subheader("Support")
    st.sidebar.markdown("""Our eResearch consultants are on hand to support your use of this app and
                        for support with data storage. For support, please contact the eResearch team
                        using UC services [Offline Transcription Feedback and Issues](https://services.canterbury.ac.nz/uc?id=sc_cat_item&sys_id=728773f587d70a10a0840649dabb3597)""")

    # Model and audio file in a form
    with st.form("setup-form", clear_on_submit=True):
        model_select = st.radio('Select a model',
                        ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                        key='model',
                        index=6,
                        horizontal=True)
        # eo = st.radio('English Only',
        #                 ['yes', 'no'],
        #                 key='eo',
        #                 index=0,
        #                 horizontal=True)
        st.session_state['eo'] = 'yes'
        max_speaker = st.number_input('Maximum Number Speaker', min_value=1, key='max_speaker',
                                      value=2, step=1)

        par_choice = st.radio("VVT or Paragraph Output",
                              ['VTT', 'paragraph'],
                              key='par',
                              index=0,
                              horizontal=True)
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload file you want to transcribe",
            accept_multiple_files=False,
        )

        transcribe_btn = st.form_submit_button("Transcribe")

    if transcribe_btn:
        if uploaded_file is not None:
            # write to temporary file and get name
            with NamedTemporaryFile() as temp:
                temp.write(uploaded_file.getvalue())
                temp.seek(0)
                audio_file = temp.name

                # Transcribe the audio file
                if st.session_state['eo'] == 'yes':
                    if st.session_state['model'].split('-')[0] == 'large':
                        model = st.session_state['model']
                    else:
                        model = st.session_state['model'] + '.en'
                else:
                    model = st.session_state['model']

                with st.spinner("Transcribing!..."):
                    RETURN_CODE = transcription(audio_file, model)

                if RETURN_CODE:
                    st.success('Transcription complete!')
                    st.session_state['disabled'] = False
                else:
                    st.warning("""Something went wrong. Please contact the eResearch Team.
                               Or try refreshing the app.""")

                with st.expander(label='Preview the transcript'):
                    st.write(st.session_state['transcript'])

    # Output widgets
    col1, col2 = st.columns(2)

    with col1:
        MIME = 'text/plain'

    with col2:
        st.write(" ")
        st.write(" ")
        with open(st.session_state['transcript_output'], 'rb') as f:
            download = st.download_button(
                        label='Download Transcript',
                        data = f,
                        file_name=st.session_state['transcript_output'].name,
                        mime = MIME,
                        disabled=st.session_state['disabled'],
                        )

            # if download:
            #     os.remove(st.session_state['transcript_output'])

    #st.write(st.session_state)
