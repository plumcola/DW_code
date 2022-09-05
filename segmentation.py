import os
from pyAudioAnalysis import audioBasicIO as aIO

import scipy.io.wavfile as wavfile
import wave
import sys

"""
A script that iterates through the extracted wav files and uses
pyAudioAnalysis' silence extraction module to make a wav file containing the
segmented audio (when the participant is speaking -- silence and virtual
interviewer speech removed)
"""

def transcript_file_processing(transcript_paths,
                               mode_for_bkgnd=False, remove_background=True):
    """
    Goes through the transcript files in the dataset and processes them in
    several ways. For the known files that contain errors, config_process,
    these are corrected. The participant and virtual agent's dialogue are
    recorded in order to remove the virtual agent in a later function. This
    also removes the background noise present at the beginning of the
    experiment unless the experiment is to solely work on this. The main
    principle in the processing is to record the onset and offset times of
    each utterance from the participant so these can be extracted from
    audio data and experimented on

    Inputs
        transcript_paths: str - The location of the transcripts
        current_dir: str - The location of the current working directory to
                     save the time signatures for each file
        mode_for_bkgnd: bool - If True, only consider the up to the
                        virtual agent's introduction, this is considered the
                        background
    remove_background: bool - Set True, if the information pre-virtual
                       agent's introduction should be removed

    Output
        on_of_times: list - Record of the participants speech time markers
                     for every file in the dataset.
    """
    on_off_times = {}
    # Interruptions during the session
    special_case =  {373: [395, 428],
             444: [286, 387]}
    # Misaligned transcript timings
    special_case_3 =  {318: 34.319917,
              321: 3.8379167,
              341: 6.1892,
              362: 16.8582}
    for i in transcript_paths:
        trial = i.split('/')[-1]
        trial = int(trial.split('_')[0])
        with open(i, 'r') as file:
            data = file.readlines()
        ellies_first_intro = 0
        inter = []
        for j, values in enumerate(data):
            file_end = len(data) - 1
            # The headers are in first position
            if j == 0:
                pass
            else:
                # The values in this list are actually strings, not int
                # Create temp which holds onset/offset times and the
                # speaker id
                temp = values.split()[0:3]
                # This corrects misalignment errors
                if trial in special_case_3:
                    if len(temp) == 0:
                        time_start = time_end = 0
                    else:
                        time_start = float(temp[0]) + special_case_3[trial]
                        time_end = float(temp[1]) + special_case_3[trial]
                else:
                    if len(temp) == 0:
                        time_start = time_end = 0
                    else:
                        time_start = float(temp[0])
                        time_end = float(temp[1])
                if len(values) > 1:
                    sync = values.split()[-1]
                else:
                    sync = ''
                if sync == '[sync]' or sync == '[syncing]':
                    sync = True
                else:
                    sync = False
                if len(temp) > 0 and temp[-1] == ('Participant' or
                                                  'participant'):
                    if sync:
                        pass
                    else:
                        if trial in special_case:
                            inter_start = special_case[trial][0]
                            inter_end = special_case[trial][1]
                            if time_start < inter_start < time_end:
                                inter.append([time_start, inter_start - 0.01])
                            elif time_start < inter_end < time_end:
                                inter.append([inter_end + 0.01, time_end])
                            elif inter_start < time_start < inter_end:
                                pass
                            elif inter_start < time_end < inter_end:
                                pass
                            elif time_end < inter_start or time_start > inter_end:
                                inter.append(temp[0:2])
                        else:
                            if 0 < j:
                                prev_val = data[j-1].split()[0:3]
                                if len(prev_val) == 0:
                                    if j - 2 > 0:
                                        prev_val = data[j-2].split()[0:3]
                                    else:
                                        prev_val = ['', '', 'Ellie']
                                if j != file_end:
                                    next_val = data[j+1].split()[0:3]
                                    if len(next_val) == 0:
                                        if j+1 != file_end:
                                            next_val = data[j + 2].split()[0:3]
                                        else:
                                            next_val = ['', '', 'Ellie']
                                else:
                                    next_val = ['', '', 'Ellie']
                                if prev_val[-1] != ('Participant' or
                                                    'participant'):
                                    holding_start = time_start
                                elif prev_val[-1] == ('Participant' or
                                                      'participant'):
                                    pass
                                if next_val[-1] == ('Participant' or
                                                    'participant'):
                                    continue
                                elif next_val[-1] != ('Participant' or
                                                      'participant'):
                                    holding_stop = time_end
                                    inter.append([str(holding_start),
                                                  str(holding_stop)])
                            else:
                                inter.append([str(time_start), str(time_end)])
                elif not temp or temp[-1] == ('Ellie' or 'ellie') and not \
                        mode_for_bkgnd and not sync:
                    pass
                elif temp[-1] == ('Ellie' or 'ellie') and mode_for_bkgnd \
                        and not sync:
                    if ellies_first_intro == 0:
                        inter.append([0, str(time_start - 0.01)])
                        break
                elif temp[-1] == ('Ellie' or 'ellie') and sync:
                    if remove_background or mode_for_bkgnd:
                        pass
                    else:
                        inter.append([str(time_start), str(time_end)])
                        ellies_first_intro = 1
                else:
                    print('Error, Transcript file does not contain '
                          'expected values')
                    print(f"File: {i}, This is from temp: {temp[-1]}")
                    sys.exit()
        on_off_times[trial]=inter

    return on_off_times

def remove_silence(filename, out_dir, on_off_times,smoothing=1.0, weight=0.3, plot=False):
    """
    A function that implements pyAudioAnalysis' silence extraction module
    and creates wav files of the participant specific portions of audio. The
    smoothing and weight parameters were tuned for the AVEC 2016 dataset.

    Parameters
    ----------
    filename : filepath
        path to the input wav file
    out_dir : filepath
        path to the desired directory (where a participant folder will
        be created containing a 'PXXX_no_silence.wav' file)
    smoothing : float
        tunable parameter to compensate for sparseness of recordings
    weight : float
        probability threshold for silence removal used in SVM
    plot : bool
        plots SVM probabilities of silence (used in tuning)

    Returns
    -------
    A folder for each participant containing a single wav file
    (named 'PXXX_no_silence.wav') with the vast majority of silence
    and virtual interviewer speech removed. Feature extraction is
    performed on these segmented wav files.
    """
    partic_id = 'P' + filename.split('/')[-1].split('_')[0]  # PXXX
    if is_segmentable(partic_id):
        print(partic_id)
        # create participant directory for segmented wav files
        participant_dir = os.path.join(out_dir, partic_id)
        if not os.path.exists(participant_dir):
            os.makedirs(participant_dir)

        os.chdir(participant_dir)
        # print(filename)
        [Fs, x] = aIO.readAudioFile(filename)

        for s in on_off_times[int(filename.split('/')[-1].split('_')[0])]:
            print(s)
            seg_name = "{:s}_{:.2f}-{:.2f}.wav".format(partic_id, float(s[0]), float(s[1]))
            wavfile.write(seg_name, Fs, x[int(Fs * float(s[0])):int(Fs * float(s[1]))])

        # concatenate segmented wave files within participant directory
        concatenate_segments(participant_dir, partic_id)


def is_segmentable(partic_id):
    """
    A function that returns True if the participant's interview clip is not
    in the manually identified set of troubled clips. The clips below were
    not segmentable do to excessive static, proximity to the virtual
    interviewer, volume levels, etc.
    """
    # troubled = set(['P300', 'P305', 'P306', 'P308', 'P315', 'P316', 'P343',
    #                 'P354', 'P362', 'P375', 'P378', 'P381', 'P382', 'P385',
    #                 'P387', 'P388', 'P390', 'P392', 'P393', 'P395', 'P408',
    #                 'P413', 'P421', 'P438', 'P473', 'P476', 'P479', 'P490',
    #                 'P492'])
    troubled = set(['P373', 'P444', 'P318', 'P321', 'P341', 'P362', 'P409',
                    'P451','P458','P480'])
    return partic_id not in troubled


def concatenate_segments(participant_dir, partic_id, remove_segment=True):
    """
    A function that concatenates all the wave files in a participants
    directory in to single wav file (with silence and other speakers removed)
    and writes in to the participant's directory, then removes the individual
    segments (when remove_segment=True).
    """
    infiles = os.listdir(participant_dir)  # list of wav files in directory
    outfile = '{}_no_silence.wav'.format(partic_id)

    data = []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
        if remove_segment:
            os.remove(infile)

    output = wave.open(outfile, 'wb')
    # details of the files must be the same (channel, frame rates, etc.)
    output.setparams(data[0][0])

    # write each segment to output
    for idx in range(len(data)):
        output.writeframes(data[idx][1])
    output.close()


if __name__ == '__main__':
    transcript_paths=[]
    tran_path="./transcripts"
    # directory containing raw wav files
    dir_name = './audio'

    # directory where a participant folder will be created containing their
    # segmented wav file
    out_dir = './interim'

    for file in os.listdir(tran_path):
        transcript_paths.append(os.path.join(tran_path,file))
    on_off_times=transcript_file_processing(transcript_paths,False,True)
    print(on_off_times.keys())
    # iterate through wav files in dir_name and create a segmented wav_file
    for file in os.listdir(dir_name):
        if file.endswith('.wav'):
            filename = os.path.join(dir_name, file)
            remove_silence(filename, out_dir,on_off_times)

