import pandas as pd
import numpy as np
import os
def smooth(csv_path,weight=0.842):
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val


    save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
    basename = os.path.basename(csv_path)
    basename = 'smooth_' + basename
    save_path = "D:\\JMCheng\\博士资料\\小论文\\Residual_Knowledge_Distillation_SE_Trans\\draw_loss_curves\\new_smoothed_curves\\" + basename
    save.to_csv(save_path)


if __name__=='__main__':


    csv_dir = "D:\\JMCheng\\博士资料\\小论文\\Residual_Knowledge_Distillation_SE_Trans\\draw_loss_curves\\new_curves"

    csv_names = []

    for dirpath, dirnames, filenames in os.walk(csv_dir):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                # print(os.path.join(dirpath,filename))
                csv_names.append(os.path.join(dirpath, filename))

    for csv_na in csv_names:
        smooth(csv_na)

