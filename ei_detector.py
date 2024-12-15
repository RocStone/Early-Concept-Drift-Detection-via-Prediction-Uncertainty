import numpy as np


class EIKMEANS():
    def __init__(self, k):
        self.residuals_chunks = [] # store the residuals of the classifier to compute test statistic, if drift detected then clear this list
        self.k = k
        self.accs = []
        self.ei_split_point = -1
        self.correct_pred = []
        self.error_chunks = []

    # @profile
    def compute_pvalue(self, dataset_name):
        # for some split ways, compute corresponding pvalue and return the minimum one
        if len(self.residuals_chunks) < 2:
            self.ei_split_point = -1
            return 1
        else:
            pvalues = []
            num_residuals = len(self.residuals_chunks)
            cut_points = [ 2**i for i in range(10) if 2**i < len(self.residuals_chunks)]
            for i in cut_points:
                left_mean = self.accs[-num_residuals:-i]
                right_mean = self.accs[-i:]
                if np.mean(self.accs[-num_residuals:-i]) < np.mean(self.accs[-i:]):
                    pvalues.append(1)
                    continue
                pvalue = self.two_sample_test(self.residuals_chunks[:-i], self.residuals_chunks[-i:], self.correct_pred[:-i], self.correct_pred[-i:], dataset_name)
                pvalues.append(pvalue)
            self.ei_split_point =  cut_points[np.argmin(pvalues)]
            return np.min(pvalues)

    def compute_pvalue_by_error(self, dataset_name):
        if len(self.error_chunks) < 2:
            self.ei_split_point = -1
            return 1
        else:
            pvalues = []
            num_residuals = len(self.error_chunks)
            cut_points = [ 2**i for i in range(10) if 2**i < len(self.error_chunks)]
            for i in cut_points:
                if np.mean(self.accs[-num_residuals:-i]) < np.mean(self.accs[-i:]):
                    pvalues.append(1)
                    continue
                pvalue = self.two_sample_test(self.error_chunks[:-i], self.error_chunks[-i:], self.correct_pred[:-i], self.correct_pred[-i:], dataset_name)
                pvalues.append(pvalue)
            self.ei_split_point =  cut_points[np.argmin(pvalues)]
            return np.min(pvalues)


    def two_sample_test(self, residuals1, residuals2, correct1, correct2, dataset_name):
        residuals1, residuals2 = np.vstack(residuals1).reshape(-1, 1), np.vstack(residuals2).reshape(-1, 1)
        correct1, correct2 = np.vstack(correct1).reshape(-1, 1), np.vstack(correct2).reshape(-1, 1)
        m_test = residuals2.shape[0]
        cp_inst = EIkMeans(self.k)
        result = cp_inst.build_partition_two_part(residuals1, m_test, correct1, dataset_name)
        if result == 1:
            pvalue = 1
            print('bug')
        else:
            if correct2.sum() == 0:
                pvalue = 0
            else:
                pvalue = cp_inst.drift_detection_two_part(residuals2, correct=correct2)
        return pvalue


from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

class EIkMeans():
    
    def __init__(self, k, lambdas=None, C=None, amplify_coe=None):
        
        self.k = k
        self.lambdas = lambdas
        self.theta = np.arange(0.0, 1.0, 0.05)
        self.C = C
        self.amplify_coe = amplify_coe
    
    def get_copy(self):
        new_copy = EIkMeans(self.k, self.lambdas, C=self.C, amplify_coe=self.amplify_coe)
        return new_copy
            
    def fill_lambda(self, data):
        

        C = self.C
        amplify_coe = self.amplify_coe
        dist_mat = euclidean_distances(C, data)
        C_idx = self.amplify_cluster(dist_mat, amplify_coe)
        k_list, unique_count = np.unique(C_idx, return_counts=True)
            
        k_list, unique_count = np.unique(C_idx, return_counts=True)
        self.lambdas = np.zeros(k_list.max()+1)
        self.lambdas[k_list.astype(int)] = unique_count

    def fill_lambda_two_part(self, data, correct):
        # data_left = data[correct].reshape(-1, 1)
        # num_data_right = data[~correct].flatten().shape[0]
        data_left = data[data<0.5].reshape(-1, 1)
        num_data_right = data[data>=0.5].flatten().shape[0]

        C = self.C
        amplify_coe = self.amplify_coe
        dist_mat = euclidean_distances(C, data_left)
        C_idx = self.amplify_cluster(dist_mat, amplify_coe)
        k_list, unique_count = np.unique(C_idx, return_counts=True)
            
        k_list, unique_count = np.unique(C_idx, return_counts=True)
        self.lambdas = np.zeros(k_list.max()+2)
        self.lambdas[k_list.astype(int)] = unique_count
        self.lambdas[-1] = num_data_right

        
    def build_partition(self, data_train, test_size):
        
        ini_size = 1000
        min_k_ratio = 0
        min_num_sample = 50
        
        if hasattr(data_train, 'shape'):
            m = data_train.shape[0]
        else:
            m = len(data_train)

        if m > ini_size:
            data_ini = data_train[:ini_size]
        else:
            data_ini = data_train
        
        if hasattr(data_ini, 'shape'):
            m_ini = data_ini.shape[0]
        else:
            m_ini = len(data_ini)
        min_5 = test_size/5
        min_50 = m/50
        min_num_p = int(np.min([min_5, min_50]))
        self.k = np.min([min_num_p, self.k])
        k = self.k
        while True:
            unique_count= [0]
            C_idx = np.zeros(m_ini)
            
            k += 1
            num_insts_part = int(np.max([m_ini/(k-1)*min_k_ratio, min_num_sample]))
            
            while np.min(unique_count) < num_insts_part:
                k -= 1
                
                if k==1:
                    # protection
                    initial_medoids = self.greed_compact_partition(data_ini, self.k)
                    kmeans = KMeans(n_clusters=self.k, n_init=1, init=data_ini[initial_medoids] ,random_state=0).fit(data_ini)
                    C = kmeans.cluster_centers_
                    amplify_coe = np.ones(self.k)
                    break
                    
                # print('num_cluster', k)
                num_insts_part = int(np.max([m_ini/(k-1)*min_k_ratio, min_num_sample]))
                
                # greedy ini
                initial_medoids = self.greed_compact_partition(data_ini, k)
                
                if np.unique(data_ini[initial_medoids].round(4)).shape[0] < k:
                    self.k = np.unique(data_ini[initial_medoids].round(4)).shape[0]
                    k = self.k + 1
                    continue
                kmeans = KMeans(n_clusters=k, n_init=1, init=data_ini[initial_medoids] ,random_state=0).fit(data_ini)
                C_idx = kmeans.labels_
                C = kmeans.cluster_centers_
                
                k_list, unique_count = np.unique(C_idx, return_counts=True)
                dr = unique_count/(m_ini/k)
                for _theta in self.theta:
                    if (dr.shape[0]) < k:
                        break
                    amplify_coe = np.exp((dr-1)*_theta)
                    C_idx = self.amplify_shrink_cluster(data_ini, C, amplify_coe)
                    k_list, unique_count = np.unique(C_idx, return_counts=True)
                    temp_unique_count = np.zeros(k)
                    temp_unique_count[k_list.astype(int)] = unique_count
                    unique_count = temp_unique_count
                    dr = unique_count/int(m_ini/k)
                    if np.min(unique_count) > num_insts_part:
                        break
            
            #===========#
            # fine tune #
            #===========#
            self.C = C
            self.amplify_coe = amplify_coe
            self.fill_lambda(data_train)
            break
        return 0

    def build_partition_two_part(self, data_train, test_size, correct_pred, dataset_name):
        
        ini_size = 1000
        min_k_ratio = 0
        min_num_sample = 50
        
        if hasattr(data_train, 'shape'):
            m = data_train.shape[0]
        else:
            m = len(data_train)

        if m > ini_size:
            data_ini_idx = np.random.choice(np.arange(data_train.flatten().shape[0]), size=ini_size)
            data_ini = data_train[data_ini_idx]
            if dataset_name not in ['pokerhand', 'insect']:
                data_ini_left = data_ini[data_ini < 0.5].reshape(-1, 1)
            else:
                correct_pred_random = correct_pred[data_ini_idx]
                data_ini_left = data_ini[correct_pred_random].reshape(-1, 1)
        else:
            # data_ini = data_trainqq
            if dataset_name not in ['pokerhand', 'insect', 'cifar10']:
                data_ini_left = data_train[data_train < 0.5].reshape(-1, 1)
            else:
                data_ini_left = data_train[correct_pred].reshape(-1, 1)


        if hasattr(data_ini_left, 'shape'):
            m_ini = data_ini_left.shape[0]
        else:
            m_ini = len(data_ini_left)
        min_5 = test_size/5
        min_50 = m/50
        min_num_p = int(np.min([min_5, min_50]))
        self.k = np.min([min_num_p, self.k])
        k = self.k
        while True:
            unique_count= [0]
            C_idx = np.zeros(m_ini)
            
            k += 1
            num_insts_part = int(np.max([m_ini/(k-1)*min_k_ratio, min_num_sample]))
            
            while np.min(unique_count) < num_insts_part:
                k -= 1
                
                if k==1:
                    # protection
                    initial_medoids = self.greed_compact_partition(data_ini_left, self.k)
                    kmeans = KMeans(n_clusters=self.k, n_init=1, init=data_ini_left[initial_medoids] ,random_state=0).fit(data_ini_left)
                    C = kmeans.cluster_centers_
                    amplify_coe = np.ones(self.k)
                    break
                    
                # print('num_cluster', k)
                num_insts_part = int(np.max([m_ini/(k-1)*min_k_ratio, min_num_sample]))
                
                # greedy ini
                initial_medoids = self.greed_compact_partition(data_ini_left, k)
                
                if np.unique(data_ini_left[initial_medoids].round(4)).shape[0] < k:
                    self.k = np.unique(data_ini_left[initial_medoids].round(4)).shape[0]
                    k = self.k + 1
                    continue
                kmeans = KMeans(n_clusters=k, n_init=1, init=data_ini_left[initial_medoids] ,random_state=0).fit(data_ini_left)
                C_idx = kmeans.labels_
                C = kmeans.cluster_centers_
                
                k_list, unique_count = np.unique(C_idx, return_counts=True)
                dr = unique_count/(m_ini/k)
                for _theta in self.theta:
                    if (dr.shape[0]) < k:
                        break
                    amplify_coe = np.exp((dr-1)*_theta)
                    C_idx = self.amplify_shrink_cluster(data_ini_left, C, amplify_coe)
                    k_list, unique_count = np.unique(C_idx, return_counts=True)
                    temp_unique_count = np.zeros(k)
                    temp_unique_count[k_list.astype(int)] = unique_count
                    unique_count = temp_unique_count
                    dr = unique_count/int(m_ini/k)
                    if np.min(unique_count) > num_insts_part:
                        break
            
            self.C = C
            self.amplify_coe = amplify_coe
            self.fill_lambda_two_part(data_train, correct_pred)
            break
        return 0


    def drift_detection_only(self, data_test, alpha):
        """detect drift with chi2 test

        Args:
            data_test (_type_): second set for detecting drift
            alpha (float): threhsold for drift detection
            beta (float): threshold for drift warning

        Returns:
            int: 0 for nothing happens, 1 for drift, 2 for warning
        """
        
        lambdas = self.lambdas
        k = len(lambdas)
        
        C = self.C
        amplify_coe = self.amplify_coe
        dist_mat = euclidean_distances(C, data_test)
        C_idx = self.amplify_cluster(dist_mat, amplify_coe)
        observations = np.zeros(k)
        
        k_list, unique_count = np.unique(C_idx, return_counts=True)
        observations[k_list.astype(int)] = unique_count
        contingency_table = np.array([lambdas, observations])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        h = 0
        if p < alpha:
            h = 1
        return h

    
    def drift_detection(self, data_test, alpha=None, beta=None):
        """detect drift with chi2 test

        Args:
            data_test (_type_): second set for detecting drift
            alpha (float): threhsold for drift detection
            beta (float): threshold for drift warning

        Returns:
            int: 0 for nothing happens, 1 for drift, 2 for warning
        """
        
        lambdas = self.lambdas
        k = len(lambdas)
        
        C = self.C
        amplify_coe = self.amplify_coe
        dist_mat = euclidean_distances(C, data_test)
        C_idx = self.amplify_cluster(dist_mat, amplify_coe)
        observations = np.zeros(k)
        
        k_list, unique_count = np.unique(C_idx, return_counts=True)
        observations[k_list.astype(int)] = unique_count
        contingency_table = np.array([lambdas, observations])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        h = 0
        return p

    def drift_detection_two_part(self, data_test, alpha=None, beta=None, correct=None):
        """detect drift with chi2 test

        Args:
            data_test (_type_): second set for detecting drift
            alpha (float): threhsold for drift detection
            beta (float): threshold for drift warning

        Returns:
            int: 0 for nothing happens, 1 for drift, 2 for warning
        """
        
        lambdas = self.lambdas
        data_test_left = data_test[correct].reshape(-1, 1)
        num_data_test_right = data_test[~correct].flatten().shape[0]
        # data_test_left = data_test[data_test<0.5].reshape(-1, 1)
        # num_data_test_right = data_test[data_test>=0.5].flatten().shape[0]

        k = len(lambdas)
        
        C = self.C
        amplify_coe = self.amplify_coe
        dist_mat = euclidean_distances(C, data_test_left)
        C_idx = self.amplify_cluster(dist_mat, amplify_coe)
        observations = np.zeros(k)
        observations[-1] = num_data_test_right
        k_list, unique_count = np.unique(C_idx, return_counts=True)
        observations[k_list.astype(int)] = unique_count

        for i in range(k):
            if observations[i] < 5:
                observations[i] = 5
            if lambdas[i] < 5:
                lambdas[i] = 5

        contingency_table = np.array([lambdas, observations])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        h = 0
        return p

    def amplify_cluster(self, dist_mat, amplify_coe, medoids_index=None):
        
        k = amplify_coe.shape[0]
        if medoids_index is None:
            C_X_dist = dist_mat
            m = dist_mat.shape[1]
        else:
            C_X_dist = dist_mat[medoids_index]
            m = dist_mat.shape[0]
        amplify_coe_mat = np.repeat(amplify_coe, m, axis=0)
        amplify_coe_mat = amplify_coe_mat.reshape(k, m)
        C_X_dist_amplified = C_X_dist*amplify_coe_mat
        np.argmin(amplify_coe_mat, axis=0)
        C_idx = np.argmin(C_X_dist_amplified, axis=0)
        return C_idx
    
    def amplify_shrink_cluster(self, data, C, amplify_coe):
        
        m = data.shape[0]
        k = C.shape[0]
        C_dist_mat = euclidean_distances(C, data)
        amplify_coe_mat = np.repeat(amplify_coe, m, axis=0)
        amplify_coe_mat = amplify_coe_mat.reshape(k, m)
        C_X_dist_amplified = C_dist_mat*amplify_coe_mat
        np.argmin(amplify_coe_mat, axis=0)
        C_idx = np.argmin(C_X_dist_amplified, axis=0)
        return C_idx
    
    def greed_compact_partition(self, data, k):
        
        if hasattr(data, 'shape'):
            m = data.shape[0]
        else:
            m = len(data)
        p_size = int(m/k)
        temp_data = np.array(data)
        C_idx = np.zeros(m) - 1
        idx_list = np.arange(m)
        for i in range(k-1):
            nbrs = NearestNeighbors(n_neighbors=p_size, algorithm='ball_tree').fit(temp_data)
            distances, indices = nbrs.kneighbors(temp_data)
            greed_idx = np.argsort(distances[:,-1])[-1]
            C_idx[idx_list[indices[greed_idx]]] = int(i)
            temp_data = np.delete(temp_data, indices[greed_idx], axis=0)
            idx_list = np.delete(idx_list, indices[greed_idx])

        C_idx[np.where(C_idx==-1)[0]] = int(k-1)
        initial_medoids = np.zeros(k) - 1
        for i in range(k):
            initial_medoids[i] = np.where(C_idx==i)[0][0]
        return initial_medoids.astype(int)
    
        