import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class KNNSampler:

    @staticmethod
    def __get_nn_scores(graph, df):
        """

        This Function Returns a NP Array of the NN_SCORES of each vertex of the KNN Graph
        NN_SCORE(xi) = | {xj | xi ∈ Nxj } |, ∀ (j != i) & (xj ∈ X)
        Nxj = Neighborhood of Xj vertex

        """

        # init nn_scores with -1
        nn_scores = np.zeros(graph.shape[0], dtype=int)
        nn_scores.fill(-1)

        # set existing vertex nn_scores as 0
        for i in range(df.shape[0]):
            nn_scores[df.index[i]] = 0

        # find nn_scores of existing vertex
        for i in range(df.shape[0]):
            # this loop runs df.shape[0] times & not graph.shape[0] times because
            # only vertex in graph that is present in df contributes to the nn_scores

            for j in range(graph.shape[1]):
                if nn_scores[graph[df.index[i]][j]] == -1:
                    continue
                else:
                    nn_scores[graph[df.index[i]][j]] += 1

        return nn_scores

    @staticmethod
    def __get_max_occurrence_indices(array):
        """

        This function returns list of indices where the maximum element of given numpy array occurs

        """

        if len(array) == 0:
            return list()
        max_list = [0]
        max_element = array[0]

        has_valid_element = False
        for i in range(len(array)):
            if array[i] != -1:
                # if is valid
                has_valid_element = True
                max_element = array[i]
                max_list[0] = i
                break

        if not has_valid_element:
            return list()

        for i in range(1, len(array)):
            if array[i] == -1:
                continue
            elif array[i] > max_element:
                max_list = [i]
                max_element = array[i]
            elif array[i] == max_element:
                max_list.append(i)
        return max_list

    @staticmethod
    def __get_max_mnn_score_indices(graph, index):
        """

        This function returns a NP Array of the MNN_SCORES of elements in index list from the graph
        MNN_SCORE(xi) = | {xj | xi ∈ Nxj ∧ xi | xj ∈ Nxi } |, ∀ (j != i) & (xj ∈ X)
        Nxj = Neighborhood of Xj vertex

        """

        mnn_scores = np.zeros(len(index))

        # calculate mnn_scores of Xi for i in index
        for i in range(len(index)):
            for j in range(graph.shape[1]):
                if np.isin(index[i], graph[graph[index[i]][j]]):
                    # add only to one of mutual neighbors to avoid + 2
                    # if 2/more mutual neighbors exits in index array itself
                    mnn_scores[i] += 1

        max_mnn_scores_indices = KNNSampler.__get_max_occurrence_indices(mnn_scores)
        # replace array index i with actual index[i] (w.r.t. dataset)
        for i in range(len(max_mnn_scores_indices)):
            max_mnn_scores_indices[i] = index[max_mnn_scores_indices[i]]
        return max_mnn_scores_indices

    @staticmethod
    def __get_row_indices_to_delete(graph, index):
        """

        This function returns a set of indices w.r.t. dataset to delete for that iteration
        Given index list, finds mutual neighbors to delete along with elements in index

        """

        # use set to avoid repetition of index
        row_indices = set()

        # add mutual neighbors
        for i in range(len(index)):
            for j in range(graph.shape[1]):
                if np.isin(index[i], graph[graph[index[i]][j]]):
                    row_indices.add(graph[index[i]][j])

        # add current index elements
        for i in range(len(index)):
            row_indices.add(index[i])

        return row_indices

    @staticmethod
    def __all_elements_zeros(nn_scores, df):
        """

        This function returns bool value
        True if all elements in nn_scores with respect to index in df have value 0
        False if there is atleast one element in nn_scores that exists in df of current iteration & its nn_score != 0

        """

        all_zeros = True
        for i in range(df.shape[0]):
            if nn_scores[df.index[i]] != 0 and nn_scores[df.index[i]] != -1:
                all_zeros = False
        return all_zeros

    @staticmethod
    def __tms_nets_selection(t, m, s, df):
        """

        This function uses TMS-NETS to select remaining rows from the dataframe df as representatives
        Basically selects rows from remaining ones after the sampling is completed

        """

        # Calculate rows per subset
        rows_per_subset = len(df) // m

        # Initialize the selected rows list
        selected_rows = []

        for i in range(m):
            if i == m - 1:
                # For the last subset, select remaining rows
                selected_rows.extend(df.sample(t - len(selected_rows)).index)
            else:
                # Select the first row randomly
                if not selected_rows:
                    selected_rows.append(np.random.choice(df.index))
                while len(selected_rows) < t:
                    # Calculate distances to selected rows
                    distances = df.drop(selected_rows).apply(
                        lambda row: np.min(df.iloc[selected_rows].apply(lambda x: np.linalg.norm(row - x), axis=1)),
                        axis=1)
                    # Select the row with the maximum distance while considering the minimum distance (s)
                    max_dist_row = distances[distances >= s].idxmax()
                    selected_rows.append(max_dist_row)

        # Get the selected rows from the DataFrame
        selected_data = df.loc[selected_rows]
        return selected_data

    @staticmethod
    def sample(X, k=5, dynamic_sampling=True, algorithm='auto', tms=(3, 1, 2)):
        """

        This function returns a new DataFrame which is a subset of given DataFrame
        The rows in this DF are samples from given DF using KNN-Sampling

        """

        # make a copy of df
        # IMPORTANT : DO NOT LET X HAVE ANY COLUMN NAMED 'idx'.
        #             IT WILL BE ADDED DURING EXECUTION FOR PROPER FUNCTIONING OF ALGORITHM

        df = pd.concat([X, pd.Series(X.index, name='idx')], axis=1).copy()

        # create empty DF to dump sampled rows
        train_samples = pd.DataFrame(columns=df.columns[:-1])

        # k + 1 neighbors because 0th neighbor is itself, which is not used in sample()
        knn_graph = NearestNeighbors(n_neighbors=k + 1, algorithm=algorithm).fit(df.iloc[:, :-1])
        _, knn_adjacency_matrix = knn_graph.kneighbors(df.iloc[:, :-1])

        # remove self neighbor column (col. number 0)
        knn_adjacency_matrix = knn_adjacency_matrix[:, 1:]

        while df.shape[0] > k:
            # get NN_SCORE
            nn_scores = KNNSampler.__get_nn_scores(knn_adjacency_matrix, df)
            if KNNSampler.__all_elements_zeros(nn_scores, df):
                break

            # shortlist indices for sampling from X using NN_SCORES
            index = KNNSampler.__get_max_occurrence_indices(nn_scores)

            # get further shortlisted index/indices to sample
            train_index = list()
            if len(index) > 1:
                train_index = train_index + KNNSampler.__get_max_mnn_score_indices(knn_adjacency_matrix, index)
            else:
                train_index = train_index + index

            # append train_index rows in train_samples
            for i in range(len(train_index)):
                train_samples = pd.concat([train_samples, df.loc[[train_index[i]]]], ignore_index=True)

            # get & remove selected indices and their mutual neighbours from X
            row_to_delete = KNNSampler.__get_row_indices_to_delete(knn_adjacency_matrix, train_index)
            try:
                df.drop(row_to_delete, inplace=True)
            except KeyError as ke:
                print("Warning Trying to delete Key that do not exist", ke.args[0])
                df.drop(row_to_delete, inplace=True, errors='ignore')

            if df.shape[0] <= k:
                # This is very important to avoid (0) len array exception in NearestNeighbors
                break

            # re-create KNN-graph if needed (Dynamic Sampling Enabled)
            if dynamic_sampling is True:
                # reset df and get new graph
                df.reset_index(inplace=True, drop=True)
                knn_graph = NearestNeighbors(n_neighbors=k + 1, algorithm=algorithm).fit(df.iloc[:, :-1])
                _, knn_adjacency_matrix = knn_graph.kneighbors(df.iloc[:, :-1])
                knn_adjacency_matrix = knn_adjacency_matrix[:, 1:]

        # leftover rows after KNN-Sampling
        if df.shape[0] != 0:
            train_samples = pd.concat([train_samples, KNNSampler.__tms_nets_selection(tms[0], tms[1], tms[2], df)], ignore_index=True)

        # return the sampled indices from X as new DF
        return train_samples
