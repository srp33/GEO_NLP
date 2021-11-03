class tester:

    def __init__(self):
        self.query_number = 0
        self.data_names = []
        self.in_file = ""

    def whichQuery(self, query_num):
        self.query_number = query_num
        self.in_file = "/Data/Queries/q{}/names.txt".format(query_num)
        with open(self.in_file, "r") as names:
            names_list = names.read()
            names = names_list.split(' ')
            self.data_names = names

    def returnScore(self, data):
        num_obtained = len(data)
        num_relevant  = 0
        total_relevant = len(self.data_names) - 1

        for name in data:
            if name in self.data_names:
                num_relevant += 1

        if num_obtained != 0 and num_relevant != 0:
            recall = num_relevant / total_relevant
            precision = num_relevant / num_obtained

            if recall != 0 and precision != 0:
                score = 2 * (precision * recall) / (precision + recall)
                return score
            else:
                return 0
        else:
            return 0

    def returnPercent(self, data):
        total_relevant = len(self.data_names) - 1
        num_relevant = 0

        for name in data:
            if name in self.data_names:
                num_relevant += 1
        return (num_relevant / total_relevant)
