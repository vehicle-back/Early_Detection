import numpy as np
import torch
import os
import pickle
import pandas as pd

from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import MinMaxScaler
from utils import seed_worker


def Extract_Certain_Range_link(link_id, Upstream_Dict, Downstream_Dict, speed_available_link_list):
    link_list = []
    if link_id in Upstream_Dict.keys():
        link_list.extend(Upstream_Dict[link_id])
    if link_id in Downstream_Dict.keys():
        link_list.extend(Downstream_Dict[link_id])
    new_link_list = [link_id]
    for link in list(set(link_list)):
        if link in speed_available_link_list:
            new_link_list.append(link)
    return new_link_list

def produce_adj_maxtrix(args, link_id_list):

    # load downstream dict
    downstream_dict = pickle.load(open(f"{args.data_dir}/data/{args.county}/processed_data/{args.county}_downstream_dict.pkl", "rb"))
    matrix_dim = len(link_id_list)
    adj_matrix = np.zeros((matrix_dim, matrix_dim))
    for i in range(matrix_dim):
        link_id = link_id_list[i]
        if  link_id in downstream_dict.keys():
            downstream_links = downstream_dict[link_id]
        else:
            continue
        for downstream_link in downstream_links:
            if downstream_link in link_id_list:
                j = link_id_list.index(downstream_link)
                adj_matrix[i, j] = 1

    return adj_matrix


def return_x_filtered_df(df, Link_List, day_point_number, number_of_points_can_be_x_perday):
    df.index = range(0, len(df))
    df['group'] = df.index // int(day_point_number)  
    df_filtered = df.groupby('group').head(number_of_points_can_be_x_perday).drop('group', axis=1)
    df_filtered = df[Link_List]
    df_filtered.fillna(0, inplace=True)
    return df_filtered

def check_nan(df):
    if df.isna().any().any():
        print(f'{df} contains nan')

def generate_link_x_y_data(args):

    link_id = args.link_id
    model_path = args.data_dir
    country_name = args.county
    upstream_range_mile = float("{:.1f}".format(args.upstream_range_mile))
    downstream_range_mile = float("{:.1f}".format(args.downstream_range_mile))
    use_sd = args.use_slowdown_spd
    use_tti = args.use_tti
    use_speed_pv = args.use_spd_pv
    use_speed_truck = args.use_spd_truck
    use_dens = args.use_dens
    use_waze_report = args.use_waze
    use_weather = args.use_weather
    use_time = args.use_time
    day_point_number = args.number_of_point_per_day
    number_of_points_can_be_x_perday = int(day_point_number-args.seq_len_out)
    number_of_points_can_be_y_perday = int(day_point_number-args.seq_len_in+1)
    inc_ahead_label_minute = args.inc_ahead_label_min

    # TMC Speed Data
    df_spd = pickle.load(open(f"{model_path}/data/{country_name}/processed_data/{country_name}_df_spd_tmc_5min_all_from_1_min.pkl", "rb"))
    df_spd_pv = pickle.load(open(f"{model_path}/data/{country_name}/processed_data/{country_name}_df_spd_tmc_5min_pv.pkl", "rb"))
    df_spd_truck = pickle.load(open(f"{model_path}/data/{country_name}/processed_data/{country_name}_df_spd_tmc_5min_truck.pkl", "rb"))  
    df_tti = pickle.load(open(f"{model_path}/data/{country_name}/processed_data/{country_name}_5min_tti.pkl", "rb"))
    df_sd_spd = pickle.load(open(f"{model_path}/data/{country_name}/processed_data/{country_name}_slowdown_speed.pkl", "rb"))
    # density data 
    df_dens = pickle.load(open(f"{model_path}/data/{country_name}/processed_data/{country_name}_df_5min_density.pkl", "rb"))
    # weather and time Data
    
    
    # incdeint data
    df_raw_waze = pickle.load(open(f"{model_path}/data/{country_name}/processed_data/{country_name}_raw_Waze.pkl", "rb"))
    # selected inc
    df_selected_inc = pickle.load(open(f"{model_path}/data/{country_name}/processed_data/{country_name}_selected_incident.pkl", "rb"))

    if inc_ahead_label_minute >0: 
        df_selected_inc_ahead_label = pickle.load(open(f"{model_path}/data/{country_name}/processed_data/{country_name}_selected_incident_ahead_{int(inc_ahead_label_minute)}.pkl", "rb"))
    
    # load useful geo data
    upstream_k_mile_dict =  pickle.load(open(f"{model_path}/data/{country_name}/processed_data/upstream_rage_dict/{country_name}_upstream_{upstream_range_mile}_mile.pkl", "rb"))
    downstream_k_mile_dict =  pickle.load(open(f"{model_path}/data/{country_name}/processed_data/downstream_rage_dict/{country_name}_downstream_{downstream_range_mile}_mile.pkl", "rb"))

    speed_available_tmc_list = list(df_spd.columns)
    Link_List = Extract_Certain_Range_link(link_id, upstream_k_mile_dict, downstream_k_mile_dict, speed_available_tmc_list)

    adj_matrix = produce_adj_maxtrix(args, Link_List)

    scaler = MinMaxScaler()

    df_extract_spd = df_spd[Link_List]
    df_extract_spd_scaled = pd.DataFrame(scaler.fit_transform(df_extract_spd), columns=df_extract_spd.columns)

    df_spd_filtered = return_x_filtered_df(df_extract_spd_scaled, Link_List, day_point_number, number_of_points_can_be_x_perday)
    Input_list = [df_spd_filtered]

    # df_raw_x = df_extract_spd_scaled


    if use_sd:
        df_extract_sd = df_sd_spd[Link_List]
        # some link may not have upstream links with aviable speed, thus slowdown speed is Nan
        df_extract_sd.fillna(0, inplace=True)
        df_extract_sd_scaled = pd.DataFrame(scaler.fit_transform(df_extract_sd), columns=df_extract_sd.columns)
        df_extract_sd_scaled = df_extract_sd_scaled[Link_List]
        df_sd_spd_filtered = return_x_filtered_df(df_extract_sd_scaled, Link_List, day_point_number, number_of_points_can_be_x_perday)
        Input_list.append(df_sd_spd_filtered)  

    if use_tti:
        df_extract_tti = df_tti[Link_List]
        df_extract_tti_scaled = pd.DataFrame(scaler.fit_transform(df_extract_tti), columns=df_extract_tti.columns)
        df_extract_tti_scaled = df_extract_tti_scaled[Link_List]
        df_tti_filtered = return_x_filtered_df(df_extract_tti_scaled, Link_List, day_point_number, number_of_points_can_be_x_perday)
        Input_list.append(df_tti_filtered)
        # df_raw_x = pd.concat([df_raw_x, df_extract_tti_scaled.reset_index(drop=True)], axis=1, ignore_index=True)


    if use_speed_pv:
        df_extract_pv_spd = df_spd_pv[[col for col in df_spd_pv.columns if col in Link_List]]
        df_extract_pv_spd_scaled = pd.DataFrame(scaler.fit_transform(df_extract_pv_spd), columns=df_extract_pv_spd.columns)
        # df_raw_x = pd.concat([df_raw_x, df_extract_pv_spd_scaled.reset_index(drop=True)], axis=1, ignore_index=True)

        zero_filled_pv_spd_df = pd.DataFrame(index=df_extract_spd_scaled.index, columns=df_extract_spd_scaled.columns)
        zero_filled_pv_spd_df.update(df_extract_pv_spd_scaled)
        zero_filled_pv_spd_df = zero_filled_pv_spd_df[Link_List]

        df_pv_spd_filtered = return_x_filtered_df(zero_filled_pv_spd_df, Link_List, day_point_number, number_of_points_can_be_x_perday)
        Input_list.append(df_pv_spd_filtered)


    if use_speed_truck:
        df_extract_truck_spd = df_spd_truck[[col for col in df_spd_truck.columns if col in Link_List]]
        df_extract_truck_spd_scaled = pd.DataFrame(scaler.fit_transform(df_extract_truck_spd), columns=df_extract_truck_spd.columns)
        # df_raw_x = pd.concat([df_raw_x, df_extract_truck_spd_scaled.reset_index(drop=True)], axis=1, ignore_index=True)

        zero_filled_truck_spd_df = pd.DataFrame(index=df_extract_spd_scaled.index, columns=df_extract_spd_scaled.columns)
        zero_filled_truck_spd_df.update(df_extract_truck_spd_scaled)
        zero_filled_truck_spd_df = zero_filled_truck_spd_df[Link_List]
        Input_list.append(zero_filled_truck_spd_df)
    
    if use_dens:
        df_extract_dens = df_dens[[col for col in df_dens.columns if col in Link_List]]
        # df_raw_x = pd.concat([df_raw_x, df_extract_dens.reset_index(drop=True)], axis=1, ignore_index=True)

        zero_filled_dens_df = pd.DataFrame(index=df_extract_spd_scaled.index, columns=df_extract_spd_scaled.columns)
        zero_filled_dens_df.update(df_extract_dens)
        
        df_dens_filtered = return_x_filtered_df(zero_filled_dens_df, Link_List, day_point_number, number_of_points_can_be_x_perday)
        Input_list.append(df_dens_filtered)


    if use_waze_report: # no need to normalize (all are 0 and 1)
        df_extract_waze = df_raw_waze[[col for col in df_raw_waze.columns if col in Link_List]]
        # df_raw_x = pd.concat([df_raw_x, df_extract_waze.reset_index(drop=True)], axis=1, ignore_index=True)
        zero_filled_waze_df = pd.DataFrame(index=df_extract_spd_scaled.index, columns=df_extract_spd_scaled.columns)
        zero_filled_waze_df.update(df_extract_waze)
        
        df_waze_filtered = return_x_filtered_df(zero_filled_waze_df, Link_List, day_point_number, number_of_points_can_be_x_perday)
        Input_list.append(df_waze_filtered)
    

    if use_weather:
        df_weather = pickle.load(open(f"{model_path}/data/{country_name}/processed_data/{country_name}_processed_weather.pkl", "rb"))
        pass
        #df_weather_scaled = pd.DataFrame(scaler.fit_transform(df_weather), columns=df_weather.columns)
        # df_raw_x = pd.concat([df_raw_x, df_weather_scaled.reset_index(drop=True)], axis=1, ignore_index=True)
    
        
    if use_time: # no need to scale
        df_time = pickle.load(open(f"{model_path}/data/{country_name}/processed_data/{country_name}_processed_time.pkl", "rb"))
        pass
        #df_raw_x = pd.concat([df_raw_x, df_time.reset_index(drop=True)], axis=1, ignore_index=True)

    df_raw_y = df_spd[[link_id]] # df_selected_inc[[link_id]] ### use 30 min info !!!!
    df_raw_y = pd.concat([df_raw_y, df_spd[[link_id]]], axis=1, join='inner')
    df_raw_y = pd.concat([df_raw_y, df_raw_waze[[link_id]]], axis=1, join='inner')

    # print(df_raw_y.shape)

    if inc_ahead_label_minute >0:
        df_raw_y = df_raw_y.reset_index(drop=True)
        df_selected_inc_ahead_label = df_selected_inc_ahead_label.reset_index(drop=True)
        df_raw_y = pd.concat([df_raw_y, df_selected_inc_ahead_label[[link_id]]], axis=1) # , join='inner'
        df_selected_inc = df_selected_inc.reset_index(drop=True)
    else:
        df_raw_y = pd.concat([df_raw_y, df_selected_inc[[link_id]]], axis=1, join='inner')
    # print(df_raw_y.shape)
    df_raw_y =  pd.concat([df_raw_y, df_selected_inc[[link_id]]], axis=1, join='inner')
    # print(df_raw_y.shape, 111, df_selected_inc[[link_id]].shape)
    
    # produce input X according to the list
    sorted_input_list = [df[Link_List] for df in Input_list]
    num_rows = sorted_input_list[0].shape[0]
    num_columns = len(Link_List)
    num_dfs = len(sorted_input_list)
    input_x = np.empty((num_rows, num_columns, num_dfs))
    
    for i in range(num_columns):
        link = Link_List[i]
        for j in range(num_dfs):
            df = sorted_input_list[j]
            input_x[:, i, j] = df[link].values


    # filter dataframe that may be used as y
    df_raw_y.index = range(0, len(df_raw_y))
    df_raw_y['group'] = df_raw_y.index // int(day_point_number)
    df_filtered_y = df_raw_y.groupby('group').tail(number_of_points_can_be_y_perday).drop('group', axis=1)



    # input_x = df_filtered_x.to_numpy()
    output_y = df_filtered_y.to_numpy()
    output_y = output_y.reshape(output_y.shape[0], 1, output_y.shape[1])

    np.save(f'{model_path}/model/link_in_out/{country_name}/GTrans_{link_id}_{upstream_range_mile}_{downstream_range_mile}_{args.seq_len_in}_{args.seq_len_out}_{str(args.use_spd_all)[0]}_{str(args.use_spd_truck)[0]}_{str(args.use_spd_pv)[0]}_{str(args.use_slowdown_spd)[0]}_{str(args.use_tti)[0]}_{str(args.use_HA)[0]}_{str(args.use_dens)[0]}_{str(args.use_weather)[0]}_{str(args.use_time)[0]}_{str(args.use_waze)[0]}_{args.inc_ahead_label_min}_x.npy', input_x)
    np.save(f'{model_path}/model/link_in_out/{country_name}/GTrans_{link_id}_{upstream_range_mile}_{downstream_range_mile}_{args.seq_len_in}_{args.seq_len_out}_{str(args.use_spd_all)[0]}_{str(args.use_spd_truck)[0]}_{str(args.use_spd_pv)[0]}_{str(args.use_slowdown_spd)[0]}_{str(args.use_tti)[0]}_{str(args.use_HA)[0]}_{str(args.use_dens)[0]}_{str(args.use_weather)[0]}_{str(args.use_time)[0]}_{str(args.use_waze)[0]}_{args.inc_ahead_label_min}_y.npy', output_y)
    np.save(f'{model_path}/model/GTrans_adj_matrix/{country_name}/{link_id}_{upstream_range_mile}_{downstream_range_mile}_adj.npy', adj_matrix)

    return input_x, output_y

class TrafficData(Dataset):
    """
    Load data under folders
    """
    def __init__(self, args):

        self.args = args     
        X, Y = generate_link_x_y_data(self.args) # in case the specifc data was not generated
        X_path = f'{args.data_dir}/model/link_in_out/{args.county}/GTrans_{args.link_id}_{float("{:.1f}".format(args.upstream_range_mile))}_{float("{:.1f}".format(args.downstream_range_mile))}_{args.seq_len_in}_{args.seq_len_out}_{str(args.use_spd_all)[0]}_{str(args.use_spd_truck)[0]}_{str(args.use_spd_pv)[0]}_{str(args.use_slowdown_spd)[0]}_{str(args.use_tti)[0]}_{str(args.use_HA)[0]}_{str(args.use_dens)[0]}_{str(args.use_weather)[0]}_{str(args.use_time)[0]}_{str(args.use_waze)[0]}_{args.inc_ahead_label_min}_x.npy'
        Y_path = f'{args.data_dir}/model/link_in_out/{args.county}/GTrans_{args.link_id}_{float("{:.1f}".format(args.upstream_range_mile))}_{float("{:.1f}".format(args.downstream_range_mile))}_{args.seq_len_in}_{args.seq_len_out}_{str(args.use_spd_all)[0]}_{str(args.use_spd_truck)[0]}_{str(args.use_spd_pv)[0]}_{str(args.use_slowdown_spd)[0]}_{str(args.use_tti)[0]}_{str(args.use_HA)[0]}_{str(args.use_dens)[0]}_{str(args.use_weather)[0]}_{str(args.use_time)[0]}_{str(args.use_waze)[0]}_{args.inc_ahead_label_min}_y.npy' 
 
        self.X = torch.from_numpy(np.load(X_path)).float()  
        self.Y = torch.from_numpy(np.load(Y_path)).float()


    def __len__(self):
        samples_per_day = self.args.number_of_point_per_day - self.args.seq_len_out - self.args.seq_len_in + 1
        return int(samples_per_day*self.args.number_of_business_day) #self.X.size(0) 

    def __getitem__(self, idx):

        number_of_points_can_be_x_perday = int(self.args.number_of_point_per_day-self.args.seq_len_out)
        number_of_points_can_be_y_perday = int(self.args.number_of_point_per_day-self.args.seq_len_in+1)
        samples_per_day = int(self.args.number_of_point_per_day - self.args.seq_len_out - self.args.seq_len_in + 1)
        day_id = idx // samples_per_day
        time_id = idx % samples_per_day
        X_idx = [day_id*number_of_points_can_be_x_perday+time_id + i for i in range(self.args.seq_len_in)]
        Y_idx = [day_id*number_of_points_can_be_y_perday+time_id + i for i in range(self.args.seq_len_out+1)]  # be careful, the starting point (first idx) of Y is the same as the last idx of X, and won't count into output sequence length
        # print('idx', idx, 'X_idx', X_idx, 'Y_idx', Y_idx)
        X = self.X[X_idx, :, :]
        Y = self.Y[Y_idx, :, :]

        return X, Y


def get_GTrans_data_loader(args):
    """
    Creates training and testing data loaders for model training
    """
    torch.manual_seed(args.seed)
    whole_dataset = TrafficData(args=args)

    # load adjency matrix
    upstream_range_mile = float("{:.1f}".format(args.upstream_range_mile))
    downstream_range_mile = float("{:.1f}".format(args.downstream_range_mile))
    adj_matrix_path = f'{args.data_dir}/model/GTrans_adj_matrix/{args.county}/{args.link_id}_{upstream_range_mile}_{downstream_range_mile}_adj.npy'
    adj_matrix = np.load(adj_matrix_path)
    

    samples_per_day = args.number_of_point_per_day - args.seq_len_out - args.seq_len_in + 1

    train_day_number = int(np.ceil(args.data_train_ratio * int(args.number_of_business_day*args.time_series_cv_ratio)))
    val_day_number = int(np.ceil(args.data_val_ratio * int(args.number_of_business_day*args.time_series_cv_ratio)))
    total_day_number = int(args.number_of_business_day*args.time_series_cv_ratio)

    day_indicies = list(range(0, int(args.number_of_business_day))) # torch.randperm(int(len(whole_dataset)/180)).tolist()
    train_day_indices = day_indicies[:train_day_number]
    val_day_indices = day_indicies[train_day_number:train_day_number+val_day_number]
    test_day_indices = day_indicies[train_day_number+val_day_number:total_day_number]

    train_indices = []
    val_indices = []
    test_indices = []


    for day_number in train_day_indices:
        single_day_indicies = [day_number*samples_per_day +time_point for time_point in range(samples_per_day)]
        selected_single_day_indicies = single_day_indicies[:len(single_day_indicies):args.training_granularity] # reduce the overfitting of highly-overlapped data
        train_indices.extend(selected_single_day_indicies)
  
    for day_number in val_day_indices:
        single_day_indicies = [day_number*samples_per_day +time_point for time_point in range(samples_per_day)]
        val_indices.extend(single_day_indicies)

    for day_number in test_day_indices:
        single_day_indicies = [day_number*samples_per_day +time_point for time_point in range(samples_per_day)]
        test_indices.extend(single_day_indicies)
    
    indices = train_indices+val_indices+test_indices

    # create subsets
    train_dataset = torch.utils.data.Subset(whole_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(whole_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(whole_dataset, test_indices)
    # np.save('train_val_test_id.npy', np.array(indices))

    train_dloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker)
    val_dloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=seed_worker)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=seed_worker)

    # read dim_in info
    x_train, _ = next(iter(train_dloader))
    model_input_dimension = x_train.size()[-1]

    # read imbalanced info
    inc_number = 0
    total_number = 0
    for _, (_, target) in enumerate(train_dloader):
        inc_number += int(target[:, :, :, 3].sum())
        total_number += target[:, :, :, 3].numel()
    imbalnaced_ratio = (total_number-inc_number)/inc_number
    
    return train_dloader, val_dloader, test_dloader, model_input_dimension, imbalnaced_ratio, adj_matrix