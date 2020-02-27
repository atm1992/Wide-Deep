# -*- coding: UTF-8 -*-

"""
train Wide & Deep model
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def get_feature_col():
    """
    age,workclass,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country
    get wide feature and deep feature
    :return:
        wide feature columns
        deep feature columns
    """
    # 连续特征
    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education-num")
    capital_gain = tf.feature_column.numeric_column("capital-gain")
    capital_loss = tf.feature_column.numeric_column("capital-loss")
    hours_per_week = tf.feature_column.numeric_column("hours-per-week")

    # 对于离散特征，先进行哈希，然后将哈希得到的结果放到wide部分；哈希后再进行embedding，得到的结果放到deep部分
    # bucket桶的大小统一设置为512，因为每一个离散特征的取值范围都不会超过512
    work_class = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=512)
    education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=512)
    marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital-status", hash_bucket_size=512)
    occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=512)
    relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=512)

    # 离散化的特征。(-inf, 18), [18, 25),..., [60, 65), [65, +inf)
    age_bucket = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    gain_bucket = tf.feature_column.bucketized_column(capital_gain, boundaries=[0, 1000, 2000, 3000, 10000])
    loss_bucket = tf.feature_column.bucketized_column(capital_loss, boundaries=[0, 1000, 2000, 3000, 5000])

    # 存储交叉特征。age_bucket分成了9段，gain_bucket分成了4段，所有9*4=36；同理 4*4=16
    cross_cols = [
        tf.feature_column.crossed_column([age_bucket, gain_bucket], hash_bucket_size=36),
        tf.feature_column.crossed_column([gain_bucket, loss_bucket], hash_bucket_size=16)
    ]
    # 存储哈希的 以及 离散化的特征
    base_cols = [work_class, education, marital_status, occupation, relationship, age_bucket, gain_bucket, loss_bucket]
    # wide部分的特征：哈希部分 + 离散化部分 + 交叉部分
    wide_cols = base_cols + cross_cols
    # deep部分特征：包含所有的连续特征 + 哈希后再embedding的特征
    # 向量的维度为9维，2^9=512，可以涵盖所有的哈希
    # deep部分输入的特征维度为5+5*9 = 50
    deep_cols = [age, education_num, capital_gain, capital_loss, hours_per_week,
                 tf.feature_column.embedding_column(work_class, 9),
                 tf.feature_column.embedding_column(education, 9),
                 tf.feature_column.embedding_column(marital_status, 9),
                 tf.feature_column.embedding_column(occupation, 9),
                 tf.feature_column.embedding_column(relationship, 9)
                 ]
    return wide_cols, deep_cols


def build_model_estimator(wide_cols, deep_cols, model_folder):
    """

    :param wide_cols: wide feature
    :param deep_cols: deep feature
    :param model_folder: origin model output folder
    :return:
        model_es 模型实例
        serving_input_func
    """
    # 4层隐含层。由上可知，deep部分输入的特征维度为5+5*9 = 50
    # 输入层(50维)与第一层隐含层(128个节点)进行全连接得到6400
    # 第一层隐含层(128个节点)与第二层隐含层(64个节点)进行全连接得到8192个参数(即 特征数)
    # 第二层隐含层(64个节点)与第三层隐含层(32个节点)进行全连接得到2048个参数(即 特征数)
    # 第三层隐含层(32个节点)与第四层隐含层(16个节点)进行全连接得到512个参数(即 特征数)
    # 6400 + 8192 + 2048 + 512 = 17152 个参数(即 特征数)
    # 因为样本数至少要为特征数的100倍，因此至少需要1715200的训练样本数
    # 但实际的训练数据只有30162条，所以需要对训练数据进行重复采样，大约57倍
    model_es = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_folder,
        linear_feature_columns=wide_cols,
        linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1, l2_regularization_strength=1.0),
        dnn_feature_columns=deep_cols,
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001,
                                                        l2_regularization_strength=0.001),
        dnn_hidden_units=[128, 64, 32, 16]
    )
    # 所有的特征
    feature_cols = wide_cols + deep_cols
    feature_spec = tf.feature_column.make_parse_example_spec(feature_cols)
    serving_input_func = (tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    return model_es, serving_input_func


def input_fn(data_file, re_time, shuffle, batch_num, predict):
    """

    :param data_file: input data, train_data or test_data
    :param re_time: 重复采样的次数，to repeat the data file，因为原数据样本太少了，样本数至少要为特征数的100倍
    :param shuffle: bool，是否要打乱数据
    :param batch_num: 采用随机梯度下降时，多少个样本之后更新一次参数
    :param predict: bool，表示是训练还是测试
    :return: 有两种情况
        若是训练，则返回train_feature, train_label
        若是测试，则返回test_feature
    """
    _CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                            [0], [0], [0], [''], ['']]
    _CSV_COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'label'
    ]

    # features返回的是一个dict，key为_CSV_COLUMNS，value为稀疏tensor(可理解为列表)
    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(list(zip(_CSV_COLUMNS, columns)))
        labels = features.pop('label')
        classes = tf.equal(labels, '>50K')  # binary classification
        return features, classes

    # 在预测时，只需返回features，而不需要返回label
    def parse_csv_predict(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(list(zip(_CSV_COLUMNS, columns)))
        labels = features.pop('label')
        return features

    # 使用TextLineDataset读取数据，过滤掉第一行列名，并过滤掉存在？的行
    data_set = tf.data.TextLineDataset(data_file).skip(1).filter(lambda line: tf.not_equal(tf.strings.regex_full_match(line, ".*\?.*"), True))
    if shuffle:
        data_set = data_set.shuffle(buffer_size=30000)
    if predict:
        data_set = data_set.map(parse_csv_predict, num_parallel_calls=5)
    else:
        data_set = data_set.map(parse_csv, num_parallel_calls=5)
    data_set = data_set.repeat(re_time)
    # 将数据分割成batch_num，用于训练或者是测试
    data_set = data_set.batch(batch_num)
    return data_set


def train_wd_model(model_es, train_file, test_file, model_export_folder, serving_input_func):
    """

    :param model_es: wide&deep模型的实例对象
    :param train_file:
    :param test_file:
    :param model_export_folder: 为提供tf serving，将模型导出到的文件夹
    :param serving_input_func: 辅助模型导出的函数
    """
    total_run = 6
    for i in range(total_run):
        # 总共运行6次，每次重复采样10倍
        model_es.train(input_fn=lambda: input_fn(train_file, re_time=10, shuffle=True, batch_num=100, predict=False))
        # 模型评估
        print("模型评估结果：", model_es.evaluate(input_fn=lambda: input_fn(test_file, re_time=1, shuffle=False, batch_num=100, predict=False)))
    # 模型导出，之所以要导出，是为了提供给tf serving搭建服务使用
    model_es.export_savedmodel(model_export_folder, serving_input_func)


def get_auc(predict_list, test_label):
    total_list = []
    for i in range(len(predict_list)):
        predict_score = predict_list[i]
        label = test_label[i]
        total_list.append((label, predict_score))
    n_pos, n_neg = 0, 0
    count = 1
    total_pos_idx = 0
    for label, _ in sorted(total_list, key=lambda ele: ele[1]):
        if label == 0:
            n_neg += 1
        else:
            n_pos += 1
            total_pos_idx += count
        count += 1
    auc_score = (total_pos_idx - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    print("auc:{:.5f}".format(auc_score))


def get_test_label(test_file):
    """
    get label of test_file
    :param test_file:
    :return:
    """
    if not os.path.exists(test_file):
        return []
    test_label_list = []
    with open(test_file, "r+") as f:
        header = next(f)
        for line in f:
            if "?" in line.strip():
                continue
            item = line.strip().split(",")
            label_str = item[-1]
            if label_str == ">50K":
                test_label_list.append(1)
            elif label_str == "<=50K":
                test_label_list.append(0)
            else:
                print("error")
    return test_label_list


def test_model_performance(model_es, test_file):
    """
    test model auc in test data
    :param model_es:
    :param test_file:
    :return:
    """
    test_label = get_test_label(test_file)
    predict_list = []
    result = model_es.predict(input_fn=lambda: input_fn(test_file, re_time=1, shuffle=False, batch_num=100, predict=True))
    for one_res in result:
        if "probabilities" in one_res:
            predict_list.append(one_res["probabilities"][1])
    get_auc(predict_list, test_label)


def run_main(train_file, test_file, model_folder, model_export_folder):
    """

    :param train_file:
    :param test_file:
    :param model_folder: origin model folder to put train model
    :param model_export_folder: for tf serving
    """
    # 决定哪些特征放到wide部分，哪些特征放到deep部分
    wide_cols, deep_cols = get_feature_col()
    # 构造模型的主体函数
    model_es, serving_input_func = build_model_estimator(wide_cols, deep_cols, model_folder)
    # 训练得到模型文件
    train_wd_model(model_es, train_file, test_file, model_export_folder, serving_input_func)
    # 利用测试数据集评估模型
    test_model_performance(model_es,test_file)


if __name__ == '__main__':
    # 注意：输入的训练文件以及测试文件中不能有空行，尤其要注意文件末尾的空行，否则会报错
    # tensorflow.python.framework.errors_impl.InvalidArgumentError: Expect 15 fields but have 0 in record 0
    run_main("../data/adult_train.txt", "../data/adult_test.txt", "../data/wide_deep", "../data/wide_deep_export")
