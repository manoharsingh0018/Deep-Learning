?	e?X???e?X???!e?X???	?Wu%z(@?Wu%z(@!?Wu%z(@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$e?X???Ș?????A#J{?/L??Y?H?}8??*	    ?_@2U
Iterator::Model::ParallelMapV2?? ?rh??!?X,??:@)?? ?rh??1?X,??:@:Preprocessing2F
Iterator::Model??z6???!?R?T*?G@)??ݓ????1?L&??d4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Zd;??!?@ 8@)-C??6??1
?B?P(4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???QI??!?P(
?6@)A??ǘ???1_????z1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipŏ1w-??!Z?V??jJ@)????Mb??1L&??d2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice-C??6z?!
?B?P(@)-C??6z?1
?B?P(@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorn??t?!???v??@)n??t?1???v??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 12.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t19.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?Wu%z(@IU?\??U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ș?????Ș?????!Ș?????      ??!       "      ??!       *      ??!       2	#J{?/L??#J{?/L??!#J{?/L??:      ??!       B      ??!       J	?H?}8???H?}8??!?H?}8??R      ??!       Z	?H?}8???H?}8??!?H?}8??b      ??!       JCPU_ONLYY?Wu%z(@b qU?\??U@Y      Y@q?q??;X@"?

both?Your program is MODERATELY input-bound because 12.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t19.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.2no:
Refer to the TF2 Profiler FAQb?96.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 