import numpy as np



class contextGeneration:
    def __init__(self,  nodes_features, test_time):
        self.nodes_features = nodes_features
        self.t = 0
        self.collected_rewards = np.array([])
        self.test_time = test_time

    def best_split(self):
        if (self.choose_split()[0] == self.no_split()[0]).all() and self.choose_split()[1] == self.no_split()[1]:
            return 0
        elif (self.choose_split()[0] == self.split_1()[0]).all() and self.choose_split()[1] == self.split_1()[1]:
            return 1
        elif (self.choose_split()[0] == self.split_2()[0]).all() and self.choose_split()[1] == self.split_2()[1]:
            return 2
        elif (self.choose_split()[0] == self.split_3()[0]).all() and self.choose_split()[1] == self.split_3()[1]:
            return 3
        elif (self.choose_split()[0] == self.split_4()[0]).all() and self.choose_split()[1] == self.split_4()[1]:
            return 4
        elif (self.choose_split()[0] == self.split_5()[0]).all() and self.choose_split()[1] == self.split_5()[1]:
            return 5
        elif (self.choose_split()[0] == self.split_6()[0]).all() and self.choose_split()[1] == self.split_6()[1]:
            return 6
        else:
            return 7


    def no_split(self):
        return np.zeros(len(self.nodes_features)), 1

    def split_1(self):
        cc = np.array([])
        for f in self.nodes_features:
            # C1 = (0,0) or (0,1)
            if f[0] == 0:
                cc = np.append(cc, 0)
            # C2 = (1,0) or (1,1)
            if f[0] == 1:
                cc = np.append(cc, 1)
        return cc, 2

    def split_2(self):
        cc = np.array([])
        for f in self.nodes_features:
            # C1 = (0,0) or (1,0)
            if f[1] == 0:
                cc = np.append(cc, 0)
            # C2 = (0,1) or (1,1)
            if f[1] == 1:
                cc = np.append(cc, 1)
        return cc, 2

    def split_3(self):
        cc = np.array([])
        for f in self.nodes_features:
            # C1 = (0,0)
            if f[0] == 0 and f[1] == 0:
                cc = np.append(cc, 0)
            # C2 = (0,1)
            if f[0] == 0 and f[1] == 1:
                cc = np.append(cc, 1)
            # C3 = (1,0) or (1,1)
            if f[0] == 1:
                cc = np.append(cc, 2)
        return cc, 3

    def split_4(self):
        cc = np.array([])
        for f in self.nodes_features:
            # C1 = (0,0) or (0,1)
            if f[0] == 0:
                cc = np.append(cc, 0)
            # C2 = (1,0)
            if f[0] == 1 and f[1] == 0:
                cc = np.append(cc, 1)
            # C3 = (1,1)
            if f[0] == 1 and f[1] == 1:
                cc = np.append(cc, 2)
        return cc, 3

    def split_5(self):
        cc = np.array([])
        for f in self.nodes_features:
            # C1 = (0,0)
            if f[0] == 0 and f[1] == 0:
                cc = np.append(cc, 0)
            # C2 = (1,0)
            if f[0] == 1 and f[1] == 0:
                cc = np.append(cc, 1)
            # C3 = (1,1)
            if f[0] == 1 and f[1] == 1:
                cc = np.append(cc, 2)
            # C4 = (0,1)
            if f[0] == 0 and f[1] == 1:
                cc = np.append(cc, 3)
        return cc, 4

    def split_6(self):
        cc = np.array([])
        for f in self.nodes_features:
            # C1 = (0,1) or (1,1)
            if f[1] == 1:
                cc = np.append(cc, 0)
            # C2 = (0,0)
            if f[0] == 0 and f[1] == 0:
                cc = np.append(cc, 1)
            # C3 = (1,0)
            if f[0] == 1 and f[1] == 0:
                cc = np.append(cc, 2)
        return cc, 3

    def split_7(self):
        cc = np.array([])
        for f in self.nodes_features:
            # C1 = (0,0) or (1,0)
            if f[1] == 0:
                cc = np.append(cc, 0)
            # C2 = (0,1)
            if f[0] == 0 and f[1] == 1:
                cc = np.append(cc, 1)
            # C3 = (1,1)
            if f[0] == 1 and f[1] == 1:
                cc = np.append(cc, 2)
        return cc, 3

    def choose_split(self):
        break_points = range(self.test_time, self.test_time * 7, self.test_time)
        if self.t < break_points[0]:
            self.t += 1
            return self.no_split()
        if break_points[0] <= self.t < break_points[1]:
            self.t += 1
            return self.split_1()
        if break_points[1] <= self.t < break_points[2]:
            self.t += 1
            return self.split_2()

        no_split_rewards = self.collected_rewards[0:break_points[0]].sum()
        split_1_rewards = self.collected_rewards[break_points[0]:break_points[1]].sum()
        split_2_rewards = self.collected_rewards[break_points[1]:break_points[2]].sum()

        if no_split_rewards > split_1_rewards and no_split_rewards > split_2_rewards:
            return self.no_split()

        if split_1_rewards > split_2_rewards:
            if break_points[2] <= self.t < break_points[3]:
                self.t += 1
                return self.split_3()
            if break_points[3] <= self.t < break_points[4]:
                self.t += 1
                return self.split_4()
            if break_points[4] <= self.t < break_points[5]:
                self.t += 1
                return self.split_5()
            split_3_rewards = self.collected_rewards[break_points[2]:break_points[3]].sum()
            split_4_rewards = self.collected_rewards[break_points[3]:break_points[4]].sum()
            split_5_rewards = self.collected_rewards[break_points[4]:break_points[5]].sum()
            if split_1_rewards > split_3_rewards and split_1_rewards > split_4_rewards and split_1_rewards > split_5_rewards:
                return self.split_1()
            if split_3_rewards > split_4_rewards and split_3_rewards > split_5_rewards:
                return self.split_3()
            if split_4_rewards > split_5_rewards:
                return self.split_4()
            else:
                return self.split_5()
        else:
            if break_points[2] <= self.t < break_points[3]:
                self.t += 1
                return self.split_6()
            if break_points[3] <= self.t < break_points[4]:
                self.t += 1
                return self.split_7()
            if break_points[4] <= self.t < break_points[5]:
                self.t += 1
                return self.split_5()
            split_6_rewards = self.collected_rewards[break_points[2]:break_points[3]].sum()
            split_7_rewards = self.collected_rewards[break_points[3]:break_points[4]].sum()
            split_5_rewards = self.collected_rewards[break_points[4]:break_points[5]].sum()
            if split_2_rewards > split_6_rewards and split_2_rewards > split_7_rewards and split_2_rewards > split_5_rewards:
                return self.split_2()
            if split_6_rewards > split_7_rewards and split_6_rewards > split_5_rewards:
                return self.split_6()
            if split_7_rewards > split_5_rewards:
                return self.split_7()
            else:
                return self.split_5()













