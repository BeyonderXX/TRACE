import datasets
import os
import json


class H_Config(datasets.BuilderConfig):
    def __init__(self,
                 *args,
                 data_file=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.data_file = data_file

class HHH(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIG_CLASS = H_Config
    BUILDER_CONFIGS = [
        H_Config(name="3H", version=VERSION, description="Plain text"),
    ]    # 对数据集的概述
    
    def _info(self):
        return datasets.DatasetInfo(
            # 这是将出现在“数据集”页面上的描述。
            # 这定义了数据集的不同列及其类型
            features=datasets.Features(
                {
                    "prompt":datasets.Value("string"),
                }
            ),     # 这一部分定义了输出关键字的类型，和要输出的关键字
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # 这个方法用来下载/提取数据，依据configurations分割数据
        # 如果可能有几种配置(在BUILDER_CONFIGS中列出)，则用户选择的配置在self.config.name中
        
        # dl_manager is a datasets.download.DownloadManager 用来下载和抽取url
        # 它可以接受任何类型或嵌套的list/dict，并将返回相同的结构，也可以将url替换为本地文件的路径。
        # 默认情况下，将提取归档文件，并返回到提取归档文件的缓存文件夹的路径，而不是归档文件

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples kwargs将会传参给_generate_examples
                gen_kwargs={
                    "filepath": self.config.data_file,
                    "split": "test",
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir["dev"]),
            #         "split": "dev",
            #     },
            # ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """
        # 这个方法将接收在前面的' _split_generators '方法中定义的' gen_kwargs '作为参数。
        # It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        # The key is not important, it's more here for legacy reason (legacy from tfds)
        # 它负责打开给定的文件并从数据集生成元组(键，示例)
        # key是不重要的，更多的是为了传承

        # 这里就是根据自己的数据集来整理
        with open(filepath, encoding="utf-8") as dataset:
            dataset=json.load(dataset)
            idx = 0
            for sample in dataset:
                prompt = sample['prompt']
                        
                yield idx,{
                    "prompt":prompt,
                }
                idx+=1
        # else:
        #     with open(filepath, encoding="utf-8") as dataset:
        #         dataset=json.load(dataset)
        #         idx=0
        #         for subgroup in dataset:
        #             subgroup_name = subgroup['sub_group']
        #             prompts = subgroup['prompts']
        #             for sample in prompts:
        #                 prompt = sample['prompt']
        #                 yield idx,{
        #                 "prompt":prompt,
        #                 "sub_group":subgroup_name
        #                 }
        #                 idx+=1
                        
                    
