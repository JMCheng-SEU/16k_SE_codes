import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import requests
import soundfile as sf

# URL for the web service
SCORING_URI = 'https://dnsmos.azurewebsites.net/score'
# If the service is authenticated, set the key or token
AUTH_KEY = 'c2V1LWVkdTpkbnNtb3M='


# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Basic {AUTH_KEY}'
def main(args):
    audio_clips_list = glob.glob(os.path.join(args.testset_dir, "*.wav"))
    scores = []
    for fpath in audio_clips_list:
        audio, fs = sf.read(fpath)
        if fs != 16000:
            print('Only sampling rate of 16000 is supported as of now')
        data = {"data": audio.tolist()}
        input_data = json.dumps(data)
        # Make the request and display the response
        resp = requests.post(SCORING_URI, data=input_data, headers=headers)
        score_dict = resp.json()
        score_dict['file_name'] = os.path.basename(fpath)
        scores.append(score_dict)

    df = pd.DataFrame(scores)
    print('Mean MOS Score for the files is ', np.mean(df['mos']))

    if args.score_file:
        df.to_csv(args.score_file)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset_dir", default="D:\\JMCheng\\DATASET\\VIVO_DATASET\\test1",
                        help='Path to the dir containing audio clips to be evaluated')
    parser.add_argument('--score_file', default="D:\\JMCheng\\DATASET\\VIVO_DATASET\\test1\\test.csv", help='If you want the scores in a CSV file provide the full path')
    args = parser.parse_args()
    main(args)