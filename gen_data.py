import os


def main():
    folder1 = '/data1/yifan.song/NTU_RGBD/nturgbd_skeletons_s001_to_s017/'
    folder2 = '/data1/yifan.song/NTU_RGBD/nturgbd_skeletons_s018_to_s032/'
    gen_ntu60(folder1)
    gen_ntu120(folder1, folder2)


def gen_ntu60(folder):
    files = os.listdir(folder)

    f_cs_train = open('./datasets/cs_train.txt','w')
    f_cv_train = open('./datasets/cv_train.txt','w')
    f_cs_eval = open('./datasets/cs_eval.txt','w')
    f_cv_eval = open('./datasets/cv_eval.txt','w')

    f_ignore = open('./datasets/ignore60.txt','r')
    ignore_names = f_ignore.readlines()
    ignore_names = [name.strip() for name in ignore_names]
    f_ignore.close()

    for file in files:
        file_name = file.split('.')[0]
        if file_name in ignore_names:
            continue
        cv = int(file_name[5:8])
        cs = int(file_name[9:12])

        if cv == 1:
            f_cv_eval.write(folder + file + '\n')
        else:
            f_cv_train.write(folder + file + '\n')

        if cs in [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]:
            f_cs_train.write(folder + file + '\n')
        else:
            f_cs_eval.write(folder + file + '\n')

    f_cs_train.close()
    f_cv_train.close()
    f_cs_eval.close()
    f_cv_eval.close()


def gen_ntu120(folder1, folder2):
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)
    f_ignore = open('./datasets/ignore120.txt','r')
    ignore_names = f_ignore.readlines()
    ignore_names = [name.strip() for name in ignore_names]
    f_ignore.close()

    f_csub_train = open('./datasets/csub_train.txt','w')
    f_cset_train = open('./datasets/cset_train.txt','w')
    f_csub_eval = open('./datasets/csub_eval.txt','w')
    f_cset_eval = open('./datasets/cset_eval.txt','w')

    write_files(files1, f_csub_train, f_cset_train, f_csub_eval, f_cset_eval, ignore_names, folder1)
    write_files(files2, f_csub_train, f_cset_train, f_csub_eval, f_cset_eval, ignore_names, folder2)

    f_csub_train.close()
    f_cset_train.close()
    f_csub_eval.close()
    f_cset_eval.close()


def write_files(files, f_csub_train, f_cset_train, f_csub_eval, f_cset_eval, ignore_names, dir_name):

    for file in files:
        file_name = file.split('.')[0]
        if not file_name in ignore_names:
            cset = int(file_name[1:4])
            csub = int(file_name[9:12])

            if cset % 2 == 0:
                f_cset_train.write(dir_name + '/' + file + '\n')
            else:
                f_cset_eval.write(dir_name + '/' + file + '\n')

            if csub in [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 
                        38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 
                        81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]:
                f_csub_train.write(dir_name + '/' + file + '\n')
            else:
                f_csub_eval.write(dir_name + '/' + file+ '\n')


if __name__ == '__main__':
    main()

