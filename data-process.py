from asyncore import read
import numpy as np
import pandas as pd
import os
import sys
import csv
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def read_c(filepath, total_pandas):
    df = pd.read_csv(filepath)
    uniq = df.name.unique()
    total_pandas = pd.concat([total_pandas, df], axis=0, ignore_index=True)
    # for col in df:
    #     print(col)
    return uniq, df

def split_by_game(df, unique):

    #create a data frame dictionary to store data frames
    DataFrameDict = {elem : pd.DataFrame() for elem in unique}
    #make a dict of dataframes split up by unique game names
    for key in DataFrameDict.keys():
        DataFrameDict[key] = df[:][df.name == key]
    return DataFrameDict

def split_by_df(dd):
    for key in dd.keys():
        dd[key].to_csv('unique_fields/' + str(key) + ".csv")

def feature_count_participants(study_dir, games_dir):
    #participant count by study
    #clear feature_counts.txt
    with open(r'C:\\Users\geoff\Documents\GitHub\Thesis-Work\feature_counts\feature_counts_study.csv', 'w') as w:
        pass
    with open(r'C:\\Users\geoff\Documents\GitHub\Thesis-Work\feature_counts\feature_counts_game.csv', 'w') as w:
        pass
    for filename in os.listdir(study_dir):
        f = os.path.join(study_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if f[-4:] == '.csv':
                # new txt file to store all feature counts
                # starting with counts of participants per study
                with open(r'C:\\Users\geoff\Documents\GitHub\Thesis-Work\feature_counts\feature_counts_study.csv', 'a', encoding='UTF8', newline='') as w:
                    writer = csv.writer(w)
                    df_pd = pd.read_csv(f)
                    # if "Participant Number" in list(df_pd.columns):
                    #     df_pd.rename(columns={'Participant Number' : 'participantNumber'}, inplace=True)
                    uniq = df_pd['participantNumber'].unique()
                    new_study = [filename[:-4], str(len(uniq))]
                    header = ['participant #', 'rows of participant', 'correct hits', 'false alarms', 'misses', 'correct rejection']
                    writer.writerow(new_study)
                    writer.writerow(header)
                    # storing number of rows per unique participant, as well as their individual hits, false alarm, miss, and correct rejection counts
                    for i in uniq:
                        data = [str(i), str(len(df_pd[df_pd['participantNumber'] == i])), str(len(df_pd[(df_pd['participantNumber'] == i) & (df_pd['interactionType'] == "correct hit")])), str(len(df_pd[(df_pd['participantNumber'] == i) & (df_pd['interactionType'] == "false alarm")])), str(len(df_pd[(df_pd['participantNumber'] == i) & (df_pd['interactionType'] == "miss")])), str(len(df_pd[(df_pd['participantNumber'] == i) & ((df_pd['interactionType'] == "correct rejection") | (df_pd['interactionType'] == "correct reject"))])), str(df_pd[(df_pd['participantNumber'] == i) & (df_pd['reactionTime'] >100)]['reactionTime'].mean())]
                        writer.writerow(data)
    for filename in os.listdir(games_dir):
        f = os.path.join(games_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if f[-4:] == '.csv':
                # new txt file to store all feature counts
                # starting with counts of participants per study
                with open(r'C:\\Users\geoff\Documents\GitHub\Thesis-Work\feature_counts\feature_counts_game.csv', 'a', encoding='UTF8', newline='') as w:
                    writer = csv.writer(w)
                    df_pd = pd.read_csv(f)
                    # if "Participant Number" in list(df_pd.columns):
                    #     df_pd.rename(columns={'Participant Number' : 'participantNumber'}, inplace=True)
                    uniq = df_pd['participantNumber'].unique()
                    new_study = [filename[:-4], str(len(uniq))]
                    header = ['participant #', 'rows of participant', 'correct hits', 'false alarms', 'misses', 'correct rejection']
                    writer.writerow(new_study)
                    writer.writerow(header)
                    # storing number of rows per unique participant, as well as their individual hits, false alarm, miss, and correct rejection counts
                    for i in uniq:
                        data = [str(i), str(len(df_pd[df_pd['participantNumber'] == i])), str(len(df_pd[(df_pd['participantNumber'] == i) & (df_pd['interactionType'] == "correct hit")])), str(len(df_pd[(df_pd['participantNumber'] == i) & (df_pd['interactionType'] == "false alarm")])), str(len(df_pd[(df_pd['participantNumber'] == i) & (df_pd['interactionType'] == "miss")])), str(len(df_pd[(df_pd['participantNumber'] == i) & ((df_pd['interactionType'] == "correct rejection") | (df_pd['interactionType'] == "correct reject"))])), str(df_pd[(df_pd['participantNumber'] == i) & (df_pd['reactionTime'] >100)]['reactionTime'].mean())]
                        writer.writerow(data)

def final_score_or_RT(game_dir):
    game_dict = {}
    for filename in os.listdir(game_dir):
        f = os.path.join(game_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if f[-4:] == '.csv':
                df_pd = pd.read_csv(f)
                #check if metric of game is final score or reaction time
                if 'reactionTime' in df_pd:
                    if (df_pd['reactionTime'] > 100).any():
                        game_dict[filename[:-4]] = 'RT'
                    else: 
                        game_dict[filename[:-4]] = 'FS'
                else:
                    game_dict[filename[:-4]] = 'FS'
    return game_dict


def game_percentile_check(game_dir, game_dict, game=None, participant_num=0, plot=False):
    if game != None and game in os.listdir(game_dir):
        f = os.path.join(game_dir, game)
        # checking if it is a file
        if os.path.isfile(f):
            if f[-4:] == '.csv' and  game_dict[game[:-4]] == 'RT':
                df_pd = pd.read_csv(f)
                uniq = df_pd['participantNumber'].unique()
                game_df = pd.DataFrame(columns=['participantNumber','avgReactionTime'])
                for i in range(0,len(uniq)):
                    game_df.loc[i] = [uniq[i]] + [df_pd[(df_pd['participantNumber'] == uniq[i]) & (df_pd['reactionTime'] >100)]['reactionTime'].mean()]
                #filter out outlier reaction times that are 3 standard deviations above the mean
                mean_rt = game_df['avgReactionTime'].mean()
                std_rt = game_df['avgReactionTime'].std()
                game_df = game_df[game_df.avgReactionTime < mean_rt + (std_rt*3)]
                if participant_num != 0 and participant_num in game_df['participantNumber']:
                    percentile_participant = stats.percentileofscore(game_df['avgReactionTime'], game_df[game_df['participantNumber'] == participant_num]['avgReactionTime'])[0]
                    print("participant " + str(participant_num) + " is better than " + str(round(percentile_participant,2)) + "% of participants")
                    percentile = str(round(percentile_participant,2))
                else:
                    print("participant number is invalid or is not present in this game")
                if plot:
                    plot_histograms(game_df, participant_num, percentile)
        else:
            print("not a valid file")
    else:
        print("no game selected")

def study_percentile_check(study_dir, game_dict, study=None, participant_num=0, plot=False):
    if study != None and study in os.listdir(study_dir):
        
        f = os.path.join(study_dir, study)
        # checking if it is a file
        if os.path.isfile(f):
            if f[-4:] == '.csv':
                df_pd = pd.read_csv(f)
                if participant_num != 0 and participant_num not in df_pd['participantNumber']:
                    print("invalid participant number")
                #build a new partial dataframe for each game in the study
                study_df = pd.DataFrame(columns=['participantNumber','avgReactionTime','gameName'])
                uniq_game = df_pd['name'].unique()
                percentile_per_game = []
                for game in uniq_game:
                    df_filtered = df_pd[df_pd['name'] == game]
                    uniq = df_filtered['participantNumber'].unique()
                    #temp_df to store a dataframe for the unique participants per game
                    temp_df = pd.DataFrame(columns=['participantNumber','avgReactionTime','gameName'])
                    for i in range(0,len(uniq)):
                        temp_df.loc[i] = [uniq[i]] + [df_filtered[(df_filtered['participantNumber'] == uniq[i]) & (df_filtered['reactionTime'] >100)]['reactionTime'].mean()] + [game]
                    #filter out outlier reaction times over 3 times the standard deviation above the mean
                    mean_rt = temp_df['avgReactionTime'].mean()
                    std_rt = temp_df['avgReactionTime'].std()
                    temp_df = temp_df[temp_df.avgReactionTime < mean_rt + (std_rt*3)]
                    if participant_num != 0 and participant_num in temp_df['participantNumber']:
                        percentile_participant = stats.percentileofscore(temp_df['avgReactionTime'], temp_df[temp_df['participantNumber'] == participant_num]['avgReactionTime'])[0]
                        percentile_per_game += [round(percentile_participant,2)]
                    study_df = study_df.append(temp_df, ignore_index=True)
                percentile = str(sum(percentile_per_game)/len(percentile_per_game))
                if plot:
                    plot(study_df, participant_num, percentile)
        else:
            print("not a valid file")
    else:
        print("no game selected")

                


def plot_histograms(df, participant_num, percentile):
    plt.hist(df['avgReactionTime'])
    plt.title("Histogram for participant #" + str(participant_num))
    plt.figtext(0.5, 0.01, "\n reaction time percentile: " + percentile + "\n", ha="left")
    plt.figtext(0.5, 0.01, "working memory metric percentile: 17.28", ha="left")
    plt.show()


if __name__ == "__main__":
    studies_directory = 'C:\\Users\\geoff\\Documents\\GitHub\\Thesis-Work\\Re__introduction_and_request_for_data'
    games_directory = 'C:\\Users\\geoff\\Documents\\GitHub\\Thesis-Work\\unique_fields'
    org_stdout = sys.stdout
    # iterate over files in that directory
    unique_games = []
    total_pandas = pd.DataFrame()
    for filename in os.listdir(studies_directory):
        f = os.path.join(studies_directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if f[-4:] == '.csv':
                # get column names for each file
                #with open(f + '.txt', 'w') as w:
                    #sys.stdout = w
                uniq, df_read = read_c(f, total_pandas)
                total_pandas = pd.concat([total_pandas, df_read], axis=0, ignore_index=True)
                unique_games.append(uniq)
                    #sys.stdout = org_stdout
    unique_games = np.unique(np.concatenate(unique_games, axis = 0))
    Data_dict = split_by_game(total_pandas, unique_games)
    split_by_df(Data_dict)
    game_dict = final_score_or_RT(games_directory)
    game_percentile_check(games_directory,game_dict, game="TagMeAgainEasy.csv", participant_num=10, plot=True)
    #study_percentile_check(studies_directory, game_dict, study="Engagement Study.csv", participant_num=1)
    #feature_count_participants(directory, games_directory)