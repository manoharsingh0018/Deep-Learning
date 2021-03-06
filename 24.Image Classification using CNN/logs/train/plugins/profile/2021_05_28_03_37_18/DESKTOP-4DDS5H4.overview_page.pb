?	?4?8Eo]@?4?8Eo]@!?4?8Eo]@	?=^??;!@?=^??;!@!?=^??;!@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?4?8Eo]@?|a2?6@AԚ??@U@Y_?L?J$@*	???????@2F
Iterator::Model?w??#9$@!?$5???X@)aTR'?)$@1dd????X@:Preprocessing2U
Iterator::Model::ParallelMapV2?sF????!?<?????)?sF????1?<?????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatT㥛? ??!?I?????)???S㥛?1?Ї`????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?#??????!???)??)??Pk?w??1?f8T?y??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceS?!?uq??!???c????)S?!?uq??1???c????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Q?????!??m? ??)?5?;Nс?1??ߵ?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;?O??nr?!_k
?	???);?O??nr?1_k
?	???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2??%䃞?!??q5Y???){?G?zd?1?Z}??$??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t19.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?=^??;!@IF8t+??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?|a2?6@?|a2?6@!?|a2?6@      ??!       "      ??!       *      ??!       2	Ԛ??@U@Ԛ??@U@!Ԛ??@U@:      ??!       B      ??!       J	_?L?J$@_?L?J$@!_?L?J$@R      ??!       Z	_?L?J$@_?L?J$@!_?L?J$@b      ??!       JCPU_ONLYY?=^??;!@b qF8t+??V@Y      Y@qho?;2@"?

both?Your program is MODERATELY input-bound because 8.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t19.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.2no:
Refer to the TF2 Profiler FAQb?18.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 