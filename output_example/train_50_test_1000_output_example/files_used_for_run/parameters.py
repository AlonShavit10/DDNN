class Parameters():
    def __init__(self):
        self.script_mode = "train" # "train" / "eval"
        self.path_for_eval = r"C:\data\git\Personal_Alon_Utils\Dictionary_Learning\data_out\check"

        self.run_name="train5_test_100_threshold20_Atoms400_Anchor2_Alpha1_Iters2000_Epoch1000_Mode1234"

        # self.run_name="conv_train5_test_100_Epoch200_100_79"

        self.base_folder = r"C:\data\git\Dict\Dictionary_Learning\data_out"

        self.ANCHOR_POINTS_STEP= 2
        self.images_train_per_class = 5
        self.images_test_per_calss = 100

        self.num_of_atoms_dict_1 = 400  # 80
        self.num_of_atoms_dict_2 = 200
        self.num_of_atoms_dict_3 = 100
        self.num_of_atoms_dict_4 = 50

        # Algorithm used to transform the data
        self.transform_algorithm = 'threshold'  # 'omp' / 'lars'/ 'threshold'/ lasso_cd
        # lars: uses the least angle regression method
        # lasso_lars: uses Lars to compute the Lasso solution
        # lasso_cd: uses the coordinate descent method to compute the Lasso solution
        # omp : uses orthogonal matching pursuit to estimate the sparse solution
        # threshold: squashes to zero all coefficients less than alpha from the projection



        self.kwargs = {'transform_alpha': 20.0}  # None/ {'transform_n_nonzero_coefs': 5} / for threshold: {'transform_alpha': .1}
        self.alpha = 1  # sparsity control. High num = more sparse
        self.n_iters = 2000  # 50 iters for learning the dict

        # DNN
        self.mode = "1234"  # "img", "1", "2", "3", "4", "12", "123", "1234"
        self.lr = 0.01 # 0.01
        self.epochs = 1000
        self.batch = 1