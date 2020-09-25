import os
class Parameters():
    def __init__(self):
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.script_mode = "eval" # "train" / "eval"
        self.path_for_eval ="output_example\train_50_test_1000_output_example"

        self.run_name=os.path.join(project_path,"train20_test_100_threshold5_Atoms200_Anchor2_Alpha2_Iters2000_Epoch1000_Mode1234_lr0.01")

        # self.run_name="conv_train20_test_100_Epoch1000_lr0.01_8_ch"

        self.base_folder = os.path.join(project_path,"data_out")

        self.images_path =os.path.join(project_path,"data\t10k-images.idx3-ubyte")
        self.labels_path = os.path.join(project_path,"data\t10k-labels.idx1-ubyte")

        self.ANCHOR_POINTS_STEP= 2
        self.IMAGES_TRAIN_PER_CLASS = 20
        self.IMAGES_TEST_PER_CLASS = 100

        self.NUM_OF_ATOMS_DICT_1 = 500  # 80
        self.NUM_OF_ATOMS_DICT_2 = 250
        self.NUM_OF_ATOMS_DICT_3 = 125
        self.NUM_OF_ATOMS_DICT_4 = 60

        self.use_another_fc=0
        self.dnn_num_ch=8

        # Algorithm used to transform the data
        self.transform_algorithm = 'threshold'  # 'omp' / 'lars'/ 'threshold'/ lasso_cd
        # lars: uses the least angle regression method
        # lasso_lars: uses Lars to compute the Lasso solution
        # lasso_cd: uses the coordinate descent method to compute the Lasso solution
        # omp : uses orthogonal matching pursuit to estimate the sparse solution
        # threshold: squashes to zero all coefficients less than alpha from the projection



        self.kwargs = {'transform_alpha': 5.0}  # None/ {'transform_n_nonzero_coefs': 5} / for threshold: {'transform_alpha': .1}
        self.ALPHA = 2  # sparsity control. High num = more sparse
        self.n_iters = 2  # 50 iters for learning the dict

        # DNN
        self.mode = "1234"  # "img", "1", "2", "3", "4", "12", "123", "1234"
        self.lr = 0.01 # 0.01
        self.epochs = 1000
        self.batch = 1