from pathlib import Path
from gc import collect

import math
import whisperx
import torch

# convert seconds to hms
def convert_to_hms(seconds: float) -> str:
    """Converts segment timestamp to hours:minuts:seconds:milliseconds"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = math.floor((seconds % 1) * 1000)
    output = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"

    return output

# convert segment to a srt like format
def convert_seg(segment):
    """Converts the segment into a string to be output into file"""
    return (f"{convert_to_hms(segment['start'])} --> {convert_to_hms(segment['end'])}\n"
            f"{'<v.'+segment['speaker']+'>' + segment['text'].lstrip()+ '</v>'}\n\n")


# convert segment to a srt like format in a paragraph
def convert_segs_par(result):
    """Converts the segments into a string to be output into file by speaker"""
    paragraph_output = {}
    speaker = result['segments'][0]['speaker']
    more_than_one = False
    decrement = 0
    paragraph = ''
    paragraph_output = {'speaker':[], 'paragraph':[]}

    for segment in result['segments']:
        try:
            if segment['speaker'] == speaker:
                paragraph += segment['text']
            else:
                more_than_one = True
                paragraph_output['speaker'].append(speaker)
                paragraph_output['paragraph'].append(paragraph)
                speaker = segment['speaker']
                paragraph = segment['text']
        except:
            decrement += 1

    if not more_than_one:
        paragraph_output['speaker'] = speaker
        paragraph_output['paragraph'] = paragraph

    return more_than_one, paragraph_output



def transcribe(audio_file_path, model, eo, par, min_speakers, max_speakers, hf_token):
    """Uses whisperx to transcribe and diarize """
    batch_size = 16
    # determine the free memory
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    # initialize the model and set the transcription to word level
    if torch.cuda.is_available() and free >= 5.0:
        device = 'cuda'
        print('running on cuda')
        if eo == 'yes':
            fw_model = whisperx.load_model(model, device, compute_type="float16", language='en')
        else:
            fw_model = whisperx.load_model(model, device, compute_type="float16")
    else:
        print('running on cpu')
        device = 'cpu'
        if eo == 'yes':
            fw_model = whisperx.load_model(model, device, compute_type="int8", language='en')
        else:
            fw_model = whisperx.load_model(model, device, compute_type="int8")
    
    # load audio file
    audio = whisperx.load_audio(audio_file_path)

    # transcribe the model
    result = fw_model.transcribe(audio, batch_size=batch_size)

    collect() 
    torch.cuda.empty_cache()
    del fw_model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # delete model if low on GPU resources
    collect()
    torch.cuda.empty_cache()
    del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

    # add min/max number of speakers if known
    min_speakers = 1
    max_speakers = 2
    diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    diarize_segments = diarize_model(audio)

    # delete model if low on GPU resources
    collect()
    torch.cuda.empty_cache()
    del diarize_model

    result = whisperx.assign_word_speakers(diarize_segments, result)

    if par=='paragaph':
        more_than_one, paragraph_output = convert_segs_par(result)
        # write the transcription file file
        tr_file_path = Path(audio_file_path + '.txt')
        with open(tr_file_path, 'w', encoding='utf-8') as tf:
            if more_than_one:
                for i, speaker in enumerate(paragraph_output['speaker']):
                    tf.write(speaker + '\n')
                    tf.write(paragraph_output['paragraph'][i].lstrip() + '\n\n')
            else:
                tf.write(paragraph_output['speaker'] + '\n')
                tf.write(paragraph_output['paragraph'].lstrip() + '\n\n')
    else:
        # write the webvtt file
        tr_file_path = Path(audio_file_path + '.vtt')
        with open(tr_file_path, 'w', encoding='utf-8') as tf:
            for i, segment in enumerate(result['segments'], start=1):
                try:
                    tf.write(f"{i-decrement}\n{convert_seg(segment)}")
                except:
                    decrement += 1

    return tr_file_path