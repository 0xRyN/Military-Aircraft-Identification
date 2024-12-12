(gpu) rayan@liliane:~/Aircraft-Identification/cnn-identification$ python3 train_script.py
2024-12-10 17:33:36.711596: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-12-10 17:33:36.724468: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1733848416.740667 63167 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1733848416.745664 63167 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-12-10 17:33:36.761571: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
{'a10': 0, 'a400m': 1, 'ag600': 2, 'ah64': 3, 'av8b': 4, 'an124': 5, 'an22': 6, 'an225': 7, 'an72': 8, 'b1': 9, 'b2': 10, 'b21': 11, 'b52': 12, 'be200': 13, 'c130': 14, 'c17': 15, 'c2': 16, 'c390': 17, 'c5': 18, 'ch47': 19, 'cl415': 20, 'e2': 21, 'e7': 22, 'ef2000': 23, 'f117': 24, 'f14': 25, 'f15': 26, 'f16': 27, 'f22': 28, 'f35': 29, 'f4': 30, 'f18': 31, 'h6': 32, 'j10': 33, 'j20': 34, 'jas39': 35, 'jf17': 36, 'jh7': 37, 'kc135': 38, 'kf21': 39, 'kj600': 40, 'ka27': 41, 'ka52': 42, 'mq9': 43, 'mi24': 44, 'mi26': 45, 'mi28': 46, 'mig29': 47, 'mig31': 48, 'mirage2000': 49, 'p3': 50, 'rq4': 51, 'rafale': 52, 'sr71': 53, 'su24': 54, 'su25': 55, 'su34': 56, 'su57': 57, 'tb001': 58, 'tb2': 59, 'tornado': 60, 'tu160': 61, 'tu22m': 62, 'tu95': 63, 'u2': 64, 'uh60': 65, 'us2': 66, 'v22': 67, 'vulcan': 68, 'wz7': 69, 'xb70': 70, 'y20': 71, 'yf23': 72, 'z19': 73}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 74/74 [00:00<00:00, 2368.18it/s]
Found 31917 aircraft images
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31917/31917 [00:00<00:00, 1873061.81it/s]
Train: 25540
Validation: 3132
Test: 3245
I0000 00:00:1733848420.441794 63167 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 46874 MB memory: -> device: 0, name: NVIDIA RTX A6000, pci bus id: 0000:3b:00.0, compute capability: 8.6
I0000 00:00:1733848420.442335 63167 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 46874 MB memory: -> device: 1, name: NVIDIA RTX A6000, pci bus id: 0000:af:00.0, compute capability: 8.6
Training Basic Model...
Model: "efficientnetb3_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type) ┃ Output Shape ┃ Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ efficientnetb3 (Functional) │ (None, 1536) │ 10,783,535 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization │ (None, 1536) │ 6,144 │
│ (BatchNormalization) │ │ │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_256 (Dense) │ (None, 256) │ 393,472 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_0.2 (Dropout) │ (None, 256) │ 0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output_layer (Dense) │ (None, 74) │ 19,018 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
Total params: 11,202,169 (42.73 MB)
Trainable params: 11,111,794 (42.39 MB)
Non-trainable params: 90,375 (353.03 KB)
Epoch 1/10
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1733848459.355378 63526 service.cc:148] XLA service 0x7530fc001cc0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1733848459.355454 63526 service.cc:156] StreamExecutor device (0): NVIDIA RTX A6000, Compute Capability 8.6
I0000 00:00:1733848459.355464 63526 service.cc:156] StreamExecutor device (1): NVIDIA RTX A6000, Compute Capability 8.6
2024-12-10 17:34:20.783872: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:268] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
I0000 00:00:1733848465.088362 63526 cuda_dnn.cc:529] Loaded cuDNN version 90300
2024-12-10 17:34:30.177354: I external/local_xla/xla/stream_executor/cuda/cuda_asm_compiler.cc:397] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_33264', 28 bytes spill stores, 28 bytes spill loads

I0000 00:00:1733848503.703854 63526 device_compiler.h:188] Compiled cluster using XLA! This line is logged at most once for the lifetime of the process.
799/799 ━━━━━━━━━━━━━━━━━━━━ 212s 165ms/step - accuracy: 0.2661 - loss: 5.8875 - val_accuracy: 0.6478 - val_loss: 1.8889
Epoch 2/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 78s 98ms/step - accuracy: 0.6983 - loss: 1.6220 - val_accuracy: 0.7047 - val_loss: 1.5243
Epoch 3/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 78s 98ms/step - accuracy: 0.7883 - loss: 1.2046 - val_accuracy: 0.7155 - val_loss: 1.5204
Epoch 4/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 79s 99ms/step - accuracy: 0.8259 - loss: 1.0582 - val_accuracy: 0.7714 - val_loss: 1.3192
Epoch 5/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 79s 99ms/step - accuracy: 0.8393 - loss: 0.9907 - val_accuracy: 0.7401 - val_loss: 1.4309
Epoch 6/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 78s 98ms/step - accuracy: 0.8661 - loss: 0.8742 - val_accuracy: 0.7640 - val_loss: 1.3292
Epoch 7/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 80s 100ms/step - accuracy: 0.8703 - loss: 0.8742 - val_accuracy: 0.7404 - val_loss: 1.5583
Epoch 8/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 79s 99ms/step - accuracy: 0.8839 - loss: 0.8236 - val_accuracy: 0.7925 - val_loss: 1.2513
Epoch 9/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 79s 98ms/step - accuracy: 0.9053 - loss: 0.6888 - val_accuracy: 0.7845 - val_loss: 1.2996
Epoch 10/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 93s 117ms/step - accuracy: 0.9040 - loss: 0.7120 - val_accuracy: 0.7727 - val_loss: 1.3477
102/102 ━━━━━━━━━━━━━━━━━━━━ 9s 85ms/step - accuracy: 0.8040 - loss: 1.2721
Basic Model saved
Training with Adamax Optimizer...
Model: "efficientnetb3_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type) ┃ Output Shape ┃ Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ efficientnetb3 (Functional) │ (None, 1536) │ 10,783,535 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization │ (None, 1536) │ 6,144 │
│ (BatchNormalization) │ │ │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_256 (Dense) │ (None, 256) │ 393,472 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_0.2 (Dropout) │ (None, 256) │ 0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output_layer (Dense) │ (None, 74) │ 19,018 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
Total params: 11,202,169 (42.73 MB)
Trainable params: 11,111,794 (42.39 MB)
Non-trainable params: 90,375 (353.03 KB)
Epoch 1/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 194s 156ms/step - accuracy: 0.2635 - loss: 6.6089 - val_accuracy: 0.7251 - val_loss: 2.4755
Epoch 2/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 83s 103ms/step - accuracy: 0.7822 - loss: 2.0155 - val_accuracy: 0.8314 - val_loss: 1.1934
Epoch 3/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 85s 106ms/step - accuracy: 0.9055 - loss: 0.8527 - val_accuracy: 0.8691 - val_loss: 0.8094
Epoch 4/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 95s 119ms/step - accuracy: 0.9460 - loss: 0.4855 - val_accuracy: 0.8748 - val_loss: 0.7068
Epoch 5/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 85s 106ms/step - accuracy: 0.9687 - loss: 0.3245 - val_accuracy: 0.8780 - val_loss: 0.6319
Epoch 6/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 95s 119ms/step - accuracy: 0.9766 - loss: 0.2487 - val_accuracy: 0.8841 - val_loss: 0.5869
Epoch 7/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 85s 107ms/step - accuracy: 0.9840 - loss: 0.1952 - val_accuracy: 0.8841 - val_loss: 0.5929
Epoch 8/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 95s 119ms/step - accuracy: 0.9865 - loss: 0.1631 - val_accuracy: 0.8994 - val_loss: 0.5093
Epoch 9/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 85s 106ms/step - accuracy: 0.9897 - loss: 0.1340 - val_accuracy: 0.8863 - val_loss: 0.5506
Epoch 10/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 78s 97ms/step - accuracy: 0.9907 - loss: 0.1233 - val_accuracy: 0.8908 - val_loss: 0.5339
102/102 ━━━━━━━━━━━━━━━━━━━━ 7s 69ms/step - accuracy: 0.8958 - loss: 0.5178
Adamax Model saved
Training with Data Augmentation...
Model: "efficientnetb3_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type) ┃ Output Shape ┃ Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ data_augmentation (Sequential) │ ? │ 0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ efficientnetb3 (Functional) │ (None, 1536) │ 10,783,535 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization │ ? │ 0 (unbuilt) │
│ (BatchNormalization) │ │ │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_256 (Dense) │ ? │ 0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_0.2 (Dropout) │ ? │ 0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output_layer (Dense) │ ? │ 0 (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
Total params: 10,783,535 (41.14 MB)
Trainable params: 10,696,232 (40.80 MB)
Non-trainable params: 87,303 (341.03 KB)
Epoch 1/10
E0000 00:00:1733850398.753631 63167 meta_optimizer.cc:966] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/efficientnetb3_model_1/efficientnetb3_1/block1b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
799/799 ━━━━━━━━━━━━━━━━━━━━ 286s 299ms/step - accuracy: 0.2184 - loss: 6.1081 - val_accuracy: 0.5763 - val_loss: 2.1647
Epoch 2/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 236s 295ms/step - accuracy: 0.6132 - loss: 2.0030 - val_accuracy: 0.6871 - val_loss: 1.6227
Epoch 3/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 236s 295ms/step - accuracy: 0.6937 - loss: 1.5814 - val_accuracy: 0.7238 - val_loss: 1.4794
Epoch 4/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 236s 295ms/step - accuracy: 0.7457 - loss: 1.4002 - val_accuracy: 0.7484 - val_loss: 1.4170
Epoch 5/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 237s 296ms/step - accuracy: 0.7688 - loss: 1.2876 - val_accuracy: 0.7589 - val_loss: 1.3478
Epoch 6/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 237s 296ms/step - accuracy: 0.7894 - loss: 1.2128 - val_accuracy: 0.7807 - val_loss: 1.2925
Epoch 7/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 234s 293ms/step - accuracy: 0.8144 - loss: 1.0806 - val_accuracy: 0.8049 - val_loss: 1.1598
Epoch 8/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 236s 295ms/step - accuracy: 0.8180 - loss: 1.0747 - val_accuracy: 0.7701 - val_loss: 1.2928
Epoch 9/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 237s 297ms/step - accuracy: 0.8422 - loss: 0.9982 - val_accuracy: 0.8011 - val_loss: 1.2532
Epoch 10/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 237s 297ms/step - accuracy: 0.8448 - loss: 0.9611 - val_accuracy: 0.8119 - val_loss: 1.1684
102/102 ━━━━━━━━━━━━━━━━━━━━ 6s 55ms/step - accuracy: 0.8088 - loss: 1.1316  
Data Augmentation Model saved
Training with Class Weights...
Model: "efficientnetb3_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type) ┃ Output Shape ┃ Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ efficientnetb3 (Functional) │ (None, 1536) │ 10,783,535 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization │ (None, 1536) │ 6,144 │
│ (BatchNormalization) │ │ │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_256 (Dense) │ (None, 256) │ 393,472 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_0.2 (Dropout) │ (None, 256) │ 0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output_layer (Dense) │ (None, 74) │ 19,018 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
Total params: 11,202,169 (42.73 MB)
Trainable params: 11,111,794 (42.39 MB)
Non-trainable params: 90,375 (353.03 KB)
Epoch 1/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 198s 153ms/step - accuracy: 0.1108 - loss: 7.1274 - val_accuracy: 0.3898 - val_loss: 3.3543
Epoch 2/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 82s 103ms/step - accuracy: 0.4657 - loss: 2.6092 - val_accuracy: 0.5230 - val_loss: 2.4799
Epoch 3/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 84s 105ms/step - accuracy: 0.6189 - loss: 1.7951 - val_accuracy: 0.5358 - val_loss: 2.4010
Epoch 4/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 96s 120ms/step - accuracy: 0.6843 - loss: 1.4833 - val_accuracy: 0.5760 - val_loss: 2.2700
Epoch 5/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 84s 105ms/step - accuracy: 0.6998 - loss: 1.4674 - val_accuracy: 0.5961 - val_loss: 2.1501
Epoch 6/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 96s 120ms/step - accuracy: 0.7117 - loss: 1.4702 - val_accuracy: 0.6967 - val_loss: 1.6469
Epoch 7/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 84s 105ms/step - accuracy: 0.8181 - loss: 0.8834 - val_accuracy: 0.6510 - val_loss: 1.9498
Epoch 8/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 96s 120ms/step - accuracy: 0.7313 - loss: 1.3817 - val_accuracy: 0.7436 - val_loss: 1.4592
Epoch 9/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 84s 105ms/step - accuracy: 0.8093 - loss: 0.9734 - val_accuracy: 0.7149 - val_loss: 1.6774
Epoch 10/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 78s 97ms/step - accuracy: 0.8151 - loss: 0.9713 - val_accuracy: 0.6724 - val_loss: 1.9704
102/102 ━━━━━━━━━━━━━━━━━━━━ 7s 71ms/step - accuracy: 0.6765 - loss: 1.9020  
Class Weights Model saved
Training Full Model with All Enhancements...
Model: "efficientnetb3_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type) ┃ Output Shape ┃ Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ data_augmentation (Sequential) │ ? │ 0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ efficientnetb3 (Functional) │ (None, 1536) │ 10,783,535 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization │ ? │ 0 (unbuilt) │
│ (BatchNormalization) │ │ │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_256 (Dense) │ ? │ 0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_0.2 (Dropout) │ ? │ 0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output_layer (Dense) │ ? │ 0 (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
Total params: 10,783,535 (41.14 MB)
Trainable params: 10,696,232 (40.80 MB)
Non-trainable params: 87,303 (341.03 KB)

Epoch 1: LearningRateScheduler setting learning rate to 0.001.
Epoch 1/10
E0000 00:00:1733853808.133640 63167 meta_optimizer.cc:966] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/efficientnetb3_model_1/efficientnetb3_1/block1b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
799/799 ━━━━━━━━━━━━━━━━━━━━ 0s 269ms/step - accuracy: 0.1229 - loss: 7.6178  
Epoch 1: val_loss improved from inf to 4.47074, saving model to best_model_sofar.keras
799/799 ━━━━━━━━━━━━━━━━━━━━ 266s 282ms/step - accuracy: 0.1231 - loss: 7.6163 - val_accuracy: 0.5112 - val_loss: 4.4707 - learning_rate: 0.0010

Epoch 2: LearningRateScheduler setting learning rate to 0.000666.
Epoch 2/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 0s 269ms/step - accuracy: 0.5055 - loss: 3.9168  
Epoch 2: val_loss improved from 4.47074 to 3.03043, saving model to best_model_sofar.keras
799/799 ━━━━━━━━━━━━━━━━━━━━ 221s 277ms/step - accuracy: 0.5055 - loss: 3.9164 - val_accuracy: 0.6753 - val_loss: 3.0304 - learning_rate: 6.6600e-04

Epoch 3: LearningRateScheduler setting learning rate to 0.00044355600000000006.
Epoch 3/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 0s 269ms/step - accuracy: 0.6712 - loss: 2.6448  
Epoch 3: val_loss improved from 3.03043 to 2.27208, saving model to best_model_sofar.keras
799/799 ━━━━━━━━━━━━━━━━━━━━ 222s 278ms/step - accuracy: 0.6712 - loss: 2.6446 - val_accuracy: 0.7548 - val_loss: 2.2721 - learning_rate: 4.4356e-04

Epoch 4: LearningRateScheduler setting learning rate to 0.00029540829600000005.
Epoch 4/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.7557 - loss: 1.9525  
Epoch 4: val_loss improved from 2.27208 to 1.86861, saving model to best_model_sofar.keras
799/799 ━━━━━━━━━━━━━━━━━━━━ 225s 282ms/step - accuracy: 0.7557 - loss: 1.9524 - val_accuracy: 0.7896 - val_loss: 1.8686 - learning_rate: 2.9541e-04

Epoch 5: LearningRateScheduler setting learning rate to 0.00019674192513600007.
Epoch 5/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 0s 270ms/step - accuracy: 0.8003 - loss: 1.5731  
Epoch 5: val_loss improved from 1.86861 to 1.60522, saving model to best_model_sofar.keras
799/799 ━━━━━━━━━━━━━━━━━━━━ 223s 279ms/step - accuracy: 0.8004 - loss: 1.5730 - val_accuracy: 0.8231 - val_loss: 1.6052 - learning_rate: 1.9674e-04

Epoch 6: LearningRateScheduler setting learning rate to 0.00013103012214057605.
Epoch 6/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 0s 267ms/step - accuracy: 0.8326 - loss: 1.3348  
Epoch 6: val_loss improved from 1.60522 to 1.46776, saving model to best_model_sofar.keras
799/799 ━━━━━━━━━━━━━━━━━━━━ 220s 275ms/step - accuracy: 0.8326 - loss: 1.3347 - val_accuracy: 0.8311 - val_loss: 1.4678 - learning_rate: 1.3103e-04

Epoch 7: LearningRateScheduler setting learning rate to 8.726606134562365e-05.
Epoch 7/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 0s 268ms/step - accuracy: 0.8536 - loss: 1.1931  
Epoch 7: val_loss improved from 1.46776 to 1.38562, saving model to best_model_sofar.keras
799/799 ━━━━━━━━━━━━━━━━━━━━ 221s 277ms/step - accuracy: 0.8536 - loss: 1.1931 - val_accuracy: 0.8436 - val_loss: 1.3856 - learning_rate: 8.7266e-05

Epoch 8: LearningRateScheduler setting learning rate to 5.811919685618535e-05.
Epoch 8/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 0s 270ms/step - accuracy: 0.8607 - loss: 1.1211  
Epoch 8: val_loss improved from 1.38562 to 1.32969, saving model to best_model_sofar.keras
799/799 ━━━━━━━━━━━━━━━━━━━━ 222s 278ms/step - accuracy: 0.8607 - loss: 1.1211 - val_accuracy: 0.8439 - val_loss: 1.3297 - learning_rate: 5.8119e-05

Epoch 9: LearningRateScheduler setting learning rate to 3.870738510621945e-05.
Epoch 9/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 0s 268ms/step - accuracy: 0.8665 - loss: 1.0660  
Epoch 9: val_loss improved from 1.32969 to 1.29170, saving model to best_model_sofar.keras
799/799 ━━━━━━━━━━━━━━━━━━━━ 221s 277ms/step - accuracy: 0.8665 - loss: 1.0659 - val_accuracy: 0.8474 - val_loss: 1.2917 - learning_rate: 3.8707e-05

Epoch 10: LearningRateScheduler setting learning rate to 2.5779118480742153e-05.
Epoch 10/10
799/799 ━━━━━━━━━━━━━━━━━━━━ 0s 269ms/step - accuracy: 0.8749 - loss: 1.0275  
Epoch 10: val_loss improved from 1.29170 to 1.26563, saving model to best_model_sofar.keras
799/799 ━━━━━━━━━━━━━━━━━━━━ 224s 280ms/step - accuracy: 0.8749 - loss: 1.0275 - val_accuracy: 0.8480 - val_loss: 1.2656 - learning_rate: 2.5779e-05
Restoring model weights from the end of the best epoch: 10.
102/102 ━━━━━━━━━━━━━━━━━━━━ 6s 55ms/step - accuracy: 0.8358 - loss: 1.2597  
Full Model saved
