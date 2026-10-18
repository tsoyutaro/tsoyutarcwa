# 同梱の参照光学データ

公開データベースrefractiveindex.infoの数値表をCSV化した。元の数値表はCC0。
https://github.com/polyanskiy/refractiveindex.info-database

- Au_Johnson_1972.csv：Johnson--Christy、室温Au、187.9--1937 nm。既定の共通Auモデル。
  原論文は真空蒸着薄膜の反射・透過から得た光学定数。
  https://doi.org/10.1103/PhysRevB.6.4370
  https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data/main/Au/nk/Johnson.yml
- Au_Yakubovsky_25nm.csv：比較用。Si上へ電子ビーム蒸着した25 nm膜、300--2000 nm。
  実際の膜厚・基材・成膜条件がMOSMITEとは異なる。30 nmの金の実測データではない。
  https://doi.org/10.1364/OE.25.025574
  https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data/main/Au/nk/Yakubovsky-25nm.yml
- PET_Zhang_400_800nm.csv：既存の未被覆MOSMITE計算と同一データ。400--800 nm、10 nm刻み。
  Chenghang plastic materialのPETに対応する参照値で、MOSMITE専用の測定値ではない。
  https://doi.org/10.1364/AO.383831
  https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data/organic/%28C10H8O4%29n%20-%20polyethylene%20terephthalate/nk/Zhang.yml

全データは正のkを使い、時間依存exp(-i omega t)に対してepsilon=(n+i k)^2とする。
各波長でn,kを別々に線形補間し、表の外側には外挿しない。
同じデータを2モデルに使うことで、最初は幾何学的な被覆形状の違いを比較する。
