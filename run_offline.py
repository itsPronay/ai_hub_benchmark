from utils.extract_metrices import extract_metrics_from_profile, log_top15_table, log_op_type_table
import json

def runOffline(path):
    with open(path, "r") as f:
        profile = json.load(f)

    metrices = extract_metrics_from_profile(profile)

    print(f"\n{'='*55}")
    print(f"  Metrics")
    print(f"{'='*55}")
    for key, value in metrices.items():
        print(f"  {key:<45} {value}")
    print(f"{'='*55}\n")

    top15_table = log_top15_table(profile)
    print(f"\n{'='*100}")
    print(f"  Top 15 Bottleneck Ops")
    print(f"{'='*100}")
    print(f"  {'Rank':<6} {'Op Name':<45} {'Type':<20} {'Unit':<6} {'Time (ms)':<12} {'%'}")
    print(f"  {'-'*93}")
    for row in top15_table.data:
        rank, name, op_type, unit, time_ms, pct = row
        print(f"  #{rank:<5} {name:<45} {op_type:<20} {unit:<6} {time_ms:<12} {pct}")
    print(f"{'='*100}\n")

    op_type_table = log_op_type_table(profile)
    print(f"\n{'='*65}")
    print(f"  Op Type Distribution")
    print(f"{'='*65}")
    print(f"  {'Op Type':<25} {'Count':<8} {'Time (ms)':<14} {'%'}")
    print(f"  {'-'*58}")
    for row in op_type_table.data:
        op_type, count, time_ms, pct = row
        print(f"  {op_type:<25} {count:<8} {time_ms:<14} {pct}")
    print(f"{'='*65}\n")

if __name__ == '__main__':
    # save profile job result in a json file and pass its location
    runOffline('job_result/ResNet_profiling_jp2me0v65_results.json')



#OUTPUT

# =======================================================
#   Metrics
# =======================================================
#   estimated_inference_time_ms                   1.464
#   mean_latency_ms                               1.5094
#   p50_latency_ms                                1.4725
#   p95_latency_ms                                1.6152
#   p99_latency_ms                                1.8659
#   min_latency_ms                                1.464
#   max_latency_ms                                3.838
#   std_dev_ms                                    0.2394
#   coeff_of_variation                            15.8608
#   throughput_fps                                662.5193
#   cold_start_ms                                 864.737
#   warm_start_ms                                 145.88
#   speedup_cold_warm                             5.9277
#   estimated_inference_peak_memory               362.9023
#   cold_start_peak_mb                            337.3516
#   warm_start_peak_mb                            393.5117
#   memory_reduction_cold_warm_pct                -16.6474
#   memory_reduction_warm_inf_pct                 7.7785
#   memory_efficiency_ratio                       1.0757
#   total_op_count                                36
#   nonzero_op_count                              32
#   zero_op_count                                 4
#   zero_op_percentage                            11.1111
#   avg_op_time_ms                                0.0895
#   total_op_time_ms                              2.864
#   dominant_compute_unit                         NPU
#   cpu_utilization_percentage                    0.0
#   gpu_utilization_percentage                    0.0
#   npu_utilization_percentage                    100.0
#   top15_ops_time_ms                             2.632
#   top15_ops_pct_of_total                        91.8994
#   effective_op_time_ratio                       0.919
# =======================================================


# ====================================================================================================
#   Top 15 Bottleneck Ops
# ====================================================================================================
#   Rank   Op Name                                       Type                 Unit   Time (ms)    %
#   ---------------------------------------------------------------------------------------------
#   #1     model/dynamic_conv2d/DynamicConv2D            CONV_2D              NPU    0.919        32.088
#   #2     model/tf.nn.max_pool2d/MaxPool2d              MAX_POOL_2D          NPU    0.55         19.2039
#   #3     model/tf.compat.v1.transpose/transpose        TRANSPOSE            NPU    0.318        11.1034
#   #4     model/dynamic_conv2d_19/DynamicConv2D         CONV_2D              NPU    0.121        4.2249
#   #5     model/dynamic_conv2d_18/DynamicConv2D         CONV_2D              NPU    0.108        3.7709
#   #6     model/dynamic_conv2d_17/DynamicConv2D         CONV_2D              NPU    0.107        3.736
#   #7     model/dynamic_conv2d_5/DynamicConv2D          CONV_2D              NPU    0.072        2.514
#   #8     model/dynamic_conv2d_13/DynamicConv2D         CONV_2D              NPU    0.066        2.3045
#   #9     model/dynamic_conv2d_14/DynamicConv2D         CONV_2D              NPU    0.065        2.2696
#   #10    model/dynamic_conv2d_12/DynamicConv2D         CONV_2D              NPU    0.064        2.2346
#   #11    model/dynamic_conv2d_10/DynamicConv2D         CONV_2D              NPU    0.057        1.9902
#   #12    model/dynamic_conv2d_15/DynamicConv2D         CONV_2D              NPU    0.056        1.9553
#   #13    model/dynamic_conv2d_8/DynamicConv2D          CONV_2D              NPU    0.044        1.5363
#   #14    model/dynamic_conv2d_9/DynamicConv2D          CONV_2D              NPU    0.043        1.5014
#   #15    model/dynamic_conv2d_7/DynamicConv2D          CONV_2D              NPU    0.042        1.4665
# ====================================================================================================


# =================================================================
#   Op Type Distribution
# =================================================================
#   Op Type                   Count    Time (ms)      %
#   ----------------------------------------------------------
#   CONV_2D                   20       1.95           68.0866
#   MAX_POOL_2D               1        0.55           19.2039
#   TRANSPOSE                 1        0.318          11.1034
#   ADD                       8        0.036          1.257
#   MEAN                      1        0.009          0.3142
#   FULLY_CONNECTED           1        0.001          0.0349
#   PAD                       4        0.0            0.0
# =================================================================