import fnmatch
import os
import zipfile


"""
A script iterates through a directory of the 189 DAIC-WOZ participant zip
files and extracts the wav and transcript files.
"""


def extract_files(zip_file, out_dir, delete_zip=False):
    """
    A function takes in a zip file and extracts the .wav file and
    *TRANSCRIPT.csv files into separate folders in a user
    specified directory.

    Parameters
    ----------
    zip_file : filepath
        path to the folder containing the DAIC-WOZ zip files
    out_dir : filepath
        path to the desired directory where audio and transcript folders
        will be created
    delete_zip : bool
        If true, deletes the zip file once relevant files are extracted

    Returns
    -------
    Two directories :
        audio : containing the extracted wav files
        transcripts : containing the extracted transcript csv files
    """
    # create audio directory
    audio_dir = os.path.join(out_dir, 'audio')
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    # create transcripts directory
    transcripts_dir = os.path.join(out_dir, 'transcripts')
    if not os.path.exists(audio_dir):
        os.makedirs(transcripts_dir)

    zip_ref = zipfile.ZipFile(zip_file)
    for f in zip_ref.namelist():  # iterate through files in zip file
        if f.endswith('.wav'):
            zip_ref.extract(f, audio_dir)
        elif fnmatch.fnmatch(f, '*TRANSCRIPT.csv'):
            zip_ref.extract(f, transcripts_dir)
    zip_ref.close()

    if delete_zip:
        os.remove(zip_file)


if __name__ == '__main__':
    # directory containing DIAC-WOZ zip files
    # dir_name = '/Volumes/Seagate Backup Plus Drive/DAIC-WOZ/'
    #
    # # directory where audio and transcripts folders will be created
    # out_dir = '/Users/zhangjian/Downloads/depression-detect/depression-detect/data/raw'
    #
    # # delete zip file after file wav and csv extraction
    # delete_zip = False
    #
    # # iterate through zip files in dir_name and extracts wav and transcripts
    # for file in os.listdir(dir_name):
    #     if file.endswith('.zip'):
    #         zip_file = os.path.join(dir_name, file)
    #         extract_files(zip_file, out_dir, delete_zip=delete_zip)
    import shutil
    path="./DAIC-WOZ"
    wav_path="./audio"
    for i in os.listdir(wav_path):
        if i==".DS_Store":
            continue
        part_id=i.split("_")[0]
        if os.path.exists(os.path.join(path,part_id+"_P")):
            continue
        os.mkdir(os.path.join(path,part_id+"_P"))
        src_path=os.path.join(wav_path,i)
        dst_path=os.path.join(path,part_id+"_P",i)
        print(src_path)
        print(dst_path)
        shutil.copyfile(src_path,dst_path)
        src_path = os.path.join(wav_path.replace("audio","transcripts"), i.replace("_AUDIO.wav","_TRANSCRIPT.csv"))
        dst_path = os.path.join(path, part_id + "_P", i.replace("_AUDIO.wav","_TRANSCRIPT.csv"))
        shutil.copyfile(src_path, dst_path)
        print(src_path)
        print(dst_path)

