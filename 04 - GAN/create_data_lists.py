from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['DIV2K\DIV2K_train_HR',
                                     'DIV2K\DIV2K_valid_HR'],
                      test_folders=['test\BSDS100',
                                    'test\Set5',
                                    'test\Set14'],
                      min_size=100,
                      output_folder='./')
