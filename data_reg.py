import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler

def load_regression_dataset(dataset_name, test_size=0.2, random_state=None, scaler="standard"):
    """
    指定した回帰ベンチマークデータセットを読み込み、前処理および学習／テスト分割を実施して返す関数。

    対応しているデータセット:
      - "abalone": Abaloneデータセット（リング数予測）
      - "boston": Boston Housingデータセット（住宅価格予測）
      - "concrete": Concrete Compressive Strength（コンクリート強度予測）
      - "energy efficiency": Energy Efficiency（建物暖房負荷予測）
      - "wine quality": Wine Quality (Red)（赤ワインの品質予測）
      - "yacht": Yacht Hydrodynamics（ヨットの残余抵抗予測）

    Parameters:
      dataset_name (str): 使用するデータセット名（大文字小文字は区別しない）
      test_size (float): テストデータの割合（default: 0.2）
      random_state (int): データ分割の乱数シード（default: 42）
      scale (bool): 特徴量をStandardScalerで標準化するか否か（default: True）

    Returns:
      X_train, X_test: 説明変数（numpy.array）
      y_train, y_test: 目的変数（numpy.array）
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'abalone':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
        columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                   'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
        df = pd.read_csv(url, header=None, names=columns)
        # カテゴリ変数「Sex」をワンホットエンコーディング
        df = pd.get_dummies(df, columns=['Sex'])
        X = df.drop('Rings', axis=1)
        y = df['Rings'].to_numpy()

    elif dataset_name in ('boston', 'boston housing'):
        # scikit-learnのfetch_openmlを利用してBostonデータを取得
        from sklearn.datasets import fetch_openml
        boston = fetch_openml(name='boston', version=1, as_frame=True)
        df = boston.frame
        # 目的変数は 'MEDV'（住宅価格）
        X = df.drop('MEDV', axis=1)
        y = df['MEDV'].to_numpy()

    elif dataset_name in ('concrete', 'concrete compressive strength'):
        import ssl

        # SSL証明書の検証を無効にする設定
        ssl._create_default_https_context = ssl._create_unverified_context

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
        df = pd.read_excel(url)

        #    Index(['Cement (component 1)(kg in a m^3 mixture)',
        #   'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
        #   'Fly Ash (component 3)(kg in a m^3 mixture)',
        #   'Water  (component 4)(kg in a m^3 mixture)',
        #   'Superplasticizer (component 5)(kg in a m^3 mixture)',
        #   'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
        #   'Fine Aggregate (component 7)(kg in a m^3 mixture)', 'Age (day)',
        #   'Concrete compressive strength(MPa, megapascals) '],
        #    dtype='object')

        # 目的変数は 'Concrete compressive strength'
        X = df.drop('Concrete compressive strength(MPa, megapascals) ', axis=1)
        y = df['Concrete compressive strength(MPa, megapascals) '].to_numpy()

    elif dataset_name in ('energy', 'energy efficiency', 'enb2012'):
        import ssl

        # SSL証明書の検証を無効にする設定
        ssl._create_default_https_context = ssl._create_unverified_context
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        df = pd.read_excel(url)
        # 最初の8列が特徴量、9列目が Heating Load、10列目が Cooling Load となっている（ここでは Heating Load を使用）
        feature_cols = df.columns[:8]
        target_col = df.columns[8]
        X = df[feature_cols]
        y = df[target_col].to_numpy()

    elif dataset_name in ('wine quality', 'wine', 'wine red'):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=';')
        # 目的変数は 'quality'
        X = df.drop('quality', axis=1)
        y = df['quality'].to_numpy()

    elif dataset_name in ('yacht', 'yacht hydrodynamics'):
        url = "https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip"
        col_names = [
            'Longitudinal_position', 'Prismatic_coefficient',
            'Length_displacement_ratio', 'Beam_draught_ratio',
            'Length_beam_ratio', 'Froude_number', 'Residuary_resistance'
        ]
        # ZIPファイルから直接読み込む（圧縮ファイル内のdataファイル名を指定）
        df = pd.read_csv(url, compression='zip', delim_whitespace=True, header=None, names=col_names)
        X = df.drop('Residuary_resistance', axis=1)
        y = df['Residuary_resistance'].to_numpy()

    elif dataset_name in ('diabetes', 'diabetes dataset'):
        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target

    elif dataset_name in ('airfoil', 'airfoil self-noise'):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
        # データは空白区切りで、6列目が目的変数（Scaled sound pressure level）
        col_names = ['Frequency', 'Angle', 'Chord_length', 'Velocity', 'Displacement_thickness', 'Sound_pressure_level']
        df = pd.read_csv(url, delim_whitespace=True, header=None, names=col_names)
        X = df.drop('Sound_pressure_level', axis=1)
        y = df['Sound_pressure_level'].to_numpy()

    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    # 学習用／テスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 特徴量の標準化
    if scaler == 'standard':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaler == 'maxabs':
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaler == 'normalizer':
        scaler = Normalizer()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaler == 'robust':
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaler == None:
        pass
    else:
        raise ValueError(f"Scaler '{scaler}' is not supported.")

    return X_train, X_test, y_train, y_test

# 例: Abaloneデータセットを読み込み
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_regression_dataset("abalone")
    print("Abaloneデータセット:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
