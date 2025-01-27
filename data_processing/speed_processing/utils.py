import pandas as pd
import pickle
import networkx as nx
import numpy as np
import geopandas as gpd
import shapely

from tqdm import tqdm
from matplotlib import pyplot as plt
######################################
#      Exploratory Data Analysis     #
######################################
# Helper function to count the number of NaN in a dataframe
def count_NaN(df):
    return sum(df.isna().sum())


def downsample_dataset(type, csv_file_path, output_file_path, set_segments, busi_date, start_time, end_time):
    df_in = pd.read_csv(csv_file_path)
    df_in.measurement_tstamp = pd.to_datetime(df_in.measurement_tstamp)
    if type == "tmc":
        df_out = df_in[
            (df_in.tmc_code.isin(set_segments)) &
            (df_in.measurement_tstamp.dt.date.isin(busi_date)) &
            (df_in.measurement_tstamp.dt.hour*60 + df_in.measurement_tstamp.dt.minute >= start_time) &
            (df_in.measurement_tstamp.dt.hour*60 + df_in.measurement_tstamp.dt.minute < end_time) 
            ]
    else:
        df_out = df_in[
            (df_in.xd_id.astype(str).isin(set_segments)) &
            (df_in.measurement_tstamp.dt.date.isin(busi_date)) &
            (df_in.measurement_tstamp.dt.hour*60 + df_in.measurement_tstamp.dt.minute >= start_time) &
            (df_in.measurement_tstamp.dt.hour*60 + df_in.measurement_tstamp.dt.minute < end_time) 
            ]
    df_out.to_csv(output_file_path, index=False)
    return df_out


def downsample_large_dataset(type, input_file_path, output_file_path, set_segments, busi_date, start_time, end_time):
    chunksize = 10 ** 7
    chunklist = []
    with pd.read_csv(input_file_path, chunksize=chunksize) as reader:
        if type == "tmc":
            for chunk in tqdm(reader):
                chunk.measurement_tstamp = pd.to_datetime(chunk.measurement_tstamp)

                # filter dataframe by selecting rows based on target segments and timestamp
                # here we select chunk from 05:30:00 to 20:59:00 to accomodate time range for both input feature (05:30:00 - 20:25:00, 5-min frequency) and output ground truth (06:00:00 - 20:59:00, 1-min frequency)
                chunk = chunk[
                        (chunk.tmc_code.isin(set_segments)) &
                        (chunk.measurement_tstamp.dt.date.isin(busi_date)) & 
                        (chunk.measurement_tstamp.dt.hour*60 + chunk.measurement_tstamp.dt.minute >= start_time) & 
                        (chunk.measurement_tstamp.dt.hour*60 + chunk.measurement_tstamp.dt.minute < end_time) 
                        ]

                chunklist.append(chunk)

        else:
            for chunk in tqdm(reader):
                chunk.measurement_tstamp = pd.to_datetime(chunk.measurement_tstamp)

                # filter dataframe by selecting rows based on target segments and timestamp
                # here we select chunk from 05:30:00 to 20:59:00 to accomodate time range for both input feature (05:30:00 - 20:25:00, 5-min frequency) and output ground truth (06:00:00 - 20:59:00, 1-min frequency)
                chunk = chunk[
                        (chunk.xd_id.astype(str).isin(set_segments)) &
                        (chunk.measurement_tstamp.dt.date.isin(busi_date)) & 
                        (chunk.measurement_tstamp.dt.hour*60 + chunk.measurement_tstamp.dt.minute >= start_time) & 
                        (chunk.measurement_tstamp.dt.hour*60 + chunk.measurement_tstamp.dt.minute < end_time) 
                        ]

                chunklist.append(chunk)

    print("Finished reading chunks!")

    # concat dataframe chunks and merge into one final dataframe 
    output = pd.concat(chunklist) 
    output = output.reset_index(drop=True)  # reset index
    output.to_csv(output_file_path, index=False)
    return output


def check_incomplete_date(df, busi_date, num_slot):
    df.index = pd.to_datetime(df.index)
    for i in busi_date:
        if df[df.index.date == i].shape[0] != num_slot:
            print(i, df[df.index.date == i].shape[0])


def pivot_df(seg_type, value_type, granularity, df, busi_date, num_slot, freq, start_time, end_time, output_file_path):
    # convert to pivot table
    if seg_type == "tmc":
        if value_type == "speed":
            df_pivot = df.pivot(index = "measurement_tstamp", columns = "tmc_code", values = "speed") 
        else:
            if "data_density" in df.columns:
                df_pivot = df.pivot(index = "measurement_tstamp", columns = "tmc_code", values = "data_density") 
            else:
                df_pivot = df.pivot(index = "measurement_tstamp", columns = "tmc_code", values = "confidence_score") 
    else:
        if value_type == "speed":
            df_pivot = df.pivot(index = "measurement_tstamp", columns = "xd_id", values = "speed") 
        else:
            df_pivot = df.pivot(index = "measurement_tstamp", columns = "xd_id", values = "confidence_score") 

    # check date with incomplete time slots
    print("Check dates of incomplete slots:")
    check_incomplete_date(df_pivot, busi_date, num_slot)

    # resample to add missing time slots
    output = df_pivot.resample(f"{freq} min").asfreq()

    # downsample to target time slots only
    output = output[
        (pd.Index(output.index.date).isin(busi_date)) &
        (output.index.hour*60 + output.index.minute >= start_time) & 
        (output.index.hour*60 + output.index.minute < end_time) 
    ]

    pickle.dump(output, open(output_file_path, "wb"))
    # return output



######################################
#           Geo Processing           #
######################################
# Helper function to compute the angle between two vectors
def angle(xd_start_lat, xd_start_long, tmc_start_lat, tmc_start_long, xd_end_lat, xd_end_long, tmc_end_lat, tmc_end_long):
    v1 = [float(xd_end_long)-float(xd_start_long), float(xd_end_lat)-float(xd_start_lat)]
    v2 = [float(tmc_end_long)-float(tmc_start_long), float(tmc_end_lat)-float(tmc_start_lat)]
    unit_v1 = v1/np.linalg.norm(v1)
    unit_v2 = v2/np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0))/np.pi


def angle_neighbor(curr_geo, prev_geo):
    '''
    Helper function to compute angle between neighbor segments
        Instead of using starting/ending lat/long of segments given in .geojson file, 
        I use the very first/last pieces of segments to more precisely compute the angle a car need to turn to travel between segments.
    '''
    # get coordinates
    if isinstance(curr_geo, shapely.geometry.linestring.LineString):  # LineString
        curr_coords = list(curr_geo.coords)
    else:  # MultiLineString
        curr_coords = list(list(curr_geo.geoms)[0].coords)
    
    if isinstance(prev_geo, shapely.geometry.linestring.LineString):  # LineString
        prev_coords = list(prev_geo.coords)
    else:  # MultiLineString
        prev_coords = list(list(prev_geo.geoms)[0].coords)

    # get coordinates of desired points
    curr_start_long, curr_start_lat = curr_coords[0] # starting position of current TMC segment: first point
    curr_end_long, curr_end_lat = curr_coords[1] # ending position of current TMC segment: second point
    prev_start_long, prev_start_lat = prev_coords[-2] # starting position of previous TMC segment: the last second point
    prev_end_long, prev_end_lat = prev_coords[-1] # ending position of previous TMC segment: the very last point

    # compute vectors and unit vectors
    curr_vec = [float(curr_end_long)-float(curr_start_long), float(curr_end_lat)-float(curr_start_lat)]
    prev_vec = [float(prev_end_long)-float(prev_start_long), float(prev_end_lat)-float(prev_start_lat)]
    unit_curr_vec = curr_vec/np.linalg.norm(curr_vec)
    unit_prev_vec = prev_vec/np.linalg.norm(prev_vec)

    # return angle in pi format
    return np.arccos(np.clip(np.dot(unit_curr_vec, unit_prev_vec), -1.0, 1.0))/np.pi


# Helper function to check direction matchness
def check_direction(d_1, d_2):
    '''
    INPUTs
        d_1: TMC direction
        d_2: XD direction
    '''
    
    wrong_directions = {'SOUTHBOUND':'NORTHBOUND', 
    'NORTHBOUND':'SOUTHBOUND', 
    'WESTBOUND':'EASTBOUND', 
    'EASTBOUND':'WESTBOUND',
    'SB': 'NB',
    'NB': 'SB',
    'WB': 'EB',
    'EB': 'WB',
    'S': "N", 
    'N': "S", 
    'W':'E', 
    'E':'W'}
    # In tmc_cranberry_v2.geojson, tmc_cranberry_v2_with_direction.geojson, tmc_shape_hwd.geojson and xd_shape_hwd_for_sjoin.geojson  there are entries in "direction" column that are not "clean" (contains road name, not denote xxxbound, etc)
    # in thoes cases, we just flag them as True for now and let it go through manual check on direction/angle in the subsequent checking procedure.

    # Naively handle cases like d_1 = "COLUMBIA GATEWAY DR SOUTHBOUND"
    for d in ["NORTHBOUND", "SOUTHBOUND", "WESTBOUND", "EASTBOUND"]:
        cnt = 0
        cand = ""
        if d in d_1:
            cand = d
            cnt += 1
    if cnt == 1:
        d_1 = cand

    if not (d_1 in wrong_directions and d_2 in wrong_directions):
        return True
    if not isinstance(d_1, str) or not isinstance(d_2, str) or d_1[0] in wrong_directions[d_2[0]]:
        return False
    else:
        return True


# Helper function to recursively extract all upstream paths starting from a source segment
# def compute_upstream(source, dict_prev, dict_upstream):
#     if source not in dict_prev:
#         # if source does not have immediate previous neighbor segment
#         return [[source]]
#     if source in dict_upstream:
#         return dict_upstream[source]
#     else:
#         # if source does have one or more immediate previous neighbor segment
#         result = []

#         # for each immediate previous neighbor p, compute the upstream of p, and prepend source into each of the upstream path
#         for p in dict_prev[source]:
#             prev_result = compute_upstream(p, dict_prev, dict_upstream)
#             for r in prev_result:
#                 result.append([source]+r)
#         return result
def compute_upstream(source, dict_prev, dict_upstream):
    if source not in dict_prev:
        # if source does not have immediate previous neighbor segment
        return [[]]
    if source in dict_upstream:
        return dict_upstream[source]
    else:
        # if source does have one or more immediate previous neighbor segment
        result = []

        # for each immediate previous neighbor p, compute the upstream of p, and prepend source into each of the upstream path
        for p in dict_prev[source]:
            prev_result = compute_upstream(p, dict_prev, dict_upstream)
            for r in prev_result:
                result.append([p]+r)
        return result


def compute_upstream_within_range(source, dict_prev, dict_upstream_within_range, dict_miles, d):
    '''
    INPUTs
        source: source segment
        dict_prev: dictionary object storing immediate upstream neighbor
            key: current segment
            value: previous neighbor(s), list or None
        dict_upstream_within_range: dictionary object storing upstream segments within range
            key: current segment
            value: upstream segment(s), set of tuple or None
        dict_miles: dictionary object storing distance of segments
            key: current segment
            value: distance in miles
        d: range

    OUTPUT
        dict_upstream_within_range: dictionary object storing upstream segments within range
            key: current segment
            value: upstream segment(s), set of tuple or None 

    '''
    if source not in dict_prev or d <= 0:
        # if source does not have immediate previous neighbor segment
        # or if the range requested is not positive
        return None

    if source in dict_upstream_within_range:
        if dict_upstream_within_range[source] is None:
            return None

        result = set()
        for upstream in dict_upstream_within_range[source]:
            r = d
            temp_result = []
            i = 0
            while r >= 0 and i < len(upstream):
                temp_result.append(upstream[i])
                r -= dict_miles[upstream[i]]
                i += 1
            result.add(tuple(temp_result))
        return result
                
    else:
        # if source does have one or more immediate previous neighbor segment
        result = set()

        # for each immediate previous neighbor p, compute the upstream of p, and prepend source into each of the upstream path
        for p in dict_prev[source]:
            if d <= dict_miles[p] or p not in dict_prev:
                result.add(tuple([p]))
            else:
                prev_result = compute_upstream_within_range(p, dict_prev, dict_upstream_within_range, dict_miles, d-dict_miles[p])
                if prev_result is None or len(prev_result) == 0: 
                    # if the upstream of previous neighbor is None or empty set
                    result.add(tuple([p]))
                else:
                    for r in prev_result:
                        if p not in r:
                            # no circles are allowed 
                            result.add(tuple([p]+list(r)))
        return result


def convert_dict_upstream_to_dict_upstream_set(dict_upstream, set_segments):
    dict_upstream_set = {}
    for seg in tqdm(list(set_segments)):
        seg_upstream_list = []
        if dict_upstream[seg] is not None:
            for upstream in dict_upstream[seg]:
                seg_upstream_list += list(upstream)
            dict_upstream_set[seg] = set(seg_upstream_list)
        else:
            dict_upstream_set[seg] = None
    return dict_upstream_set


'''
obsolete helper functions to compute upstream and downstream tmc segments (as proposed by Weiran)
'''
# def read_shp(path):
#     net = nx.read_shp(path, simplify=True)
#     return net

# def approx_mile(tmc2edge, tmc_id):
#     d = 0.0
#     for end, begin in tmc2edge[tmc_id]:
#         temp = ((begin[0]-end[0])**2 + (begin[1]-end[1])**2)**(0.5)
#         d += temp
#     # d /= len(tmc2edge[tmc_id])
#     return d

# def build_tmc_dict(net):
#     tmc2edge = { }
#     for key in net.edges:
#         tmc_id = net.edges[key]['id']
#         if tmc_id not in tmc2edge:
#             tmc2edge[tmc_id] = [ ]
#         tmc2edge[tmc_id].append(key)
#     return tmc2edge

# def get_upstream_tmc(tmc2edge, tmc_id, tmc_attr, net, wrong_directions, d=5):
#     out = set() 
#     wrong_direct = wrong_directions[tmc_attr[tmc_attr.id_tmc==tmc_id].direction.values[0]]   
#     for (end_node, bgn_node) in tmc2edge[tmc_id]:
#         dist = d
#         # for e in nx.bfs_predecessors(net, bgn_node):
#         for e in nx.bfs_edges(net, bgn_node, reverse=True):
#             # print(e, bgn_node, end_node)
#             prev_tmc = net.edges[e]['id']

#             # avoid repeating segments or tmc_id itself
#             if prev_tmc == tmc_id or prev_tmc in out:
#                 continue
#             else:
#                 # compute the range of nxt_tmc segment
#                 degree = 0
#                 if prev_tmc in set(tmc_attr.id_tmc.unique()):
#                     degree = angle(xd_start_lat=e[1][1], xd_start_long=e[1][0], tmc_start_lat=bgn_node[1], tmc_start_long=bgn_node[0], xd_end_lat=e[0][1], xd_end_long=e[0][0], tmc_end_lat=end_node[1], tmc_end_long=end_node[0])
#                     if degree < 0.5 and tmc_attr[tmc_attr.id_tmc==prev_tmc].direction.values[0] != wrong_direct:
#                         dist -= tmc_attr[tmc_attr.id_tmc==prev_tmc].miles_tmc.values[0]
#                         out.add(prev_tmc)
#                 # else:
#                 #     # if there's no record of range of prev_tmc, then compute the distance from the beginning point to the end point of prev_tmc
#                 #     dist -= approx_mile(tmc2edge, prev_tmc)
                

#             if dist <= 0:
#                 break    
#     return list(out)

# def get_downstream_tmc(tmc2edge, tmc_id, miles_tmc, net, d=5):
#     out = set()    
#     for (end_node, bgn_node) in tmc2edge[tmc_id]:
#         dist = d
#         for e in nx.bfs_edges(net, end_node, reverse=True):
#             # reverve back due to reverse of graph in bfs
#             r_e = (e[1],e[0])
#             nxt_tmc = net.edges[r_e]['id']
            
#             # avoid repeating segments or tmc_id itself
#             if nxt_tmc == tmc_id or nxt_tmc in out:
#                 continue
#             else:
#                 out.add(nxt_tmc)

#                 # compute the range of nxt_tmc segment
#                 if nxt_tmc in miles_tmc:
#                     dist -= miles_tmc[nxt_tmc]["miles_tmc"]
#                 # else:
#                 #     dist -= approx_mile(tmc2edge, nxt_tmc)
                

#             if dist <= 0:
#                 break    
#     return list(out)

# def get_neighbor_tmc(tmc_id, net, n=5):
#     tmc2edge = build_tmc_dict(net)
#     up = get_upstream_tmc(tmc2edge, tmc_id, net, n)
#     dn = get_downstream_tmc(tmc2edge, tmc_id, net, n)
#     return np.concatenate((up, dn))