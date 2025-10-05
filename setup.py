import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="nnunetv2",  # 项目名称
        version="2.0",  # 项目版本
        packages=["nnunetv2"],  # 显式指定包名
        install_requires=[
            "numpy",  # 添加必要的依赖包
            "torch",
        ],
    )