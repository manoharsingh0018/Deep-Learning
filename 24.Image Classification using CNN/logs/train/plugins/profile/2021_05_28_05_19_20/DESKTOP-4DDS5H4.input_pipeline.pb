	?_vO?9@?_vO?9@!?_vO?9@	޵??e???޵??e???!޵??e???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?_vO?9@Dio?????A?=yX?9@Y?W?2??*	?????9_@2U
Iterator::Model::ParallelMapV2???~?:??!?o?[a9@)???~?:??1?o?[a9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??~j?t??!?????l>@)? ?	???1?????8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX9??v???!?@D??8@)?A`??"??1$P???75@:Preprocessing2F
Iterator::ModelbX9?Ȧ?!T???r?A@)-C??6??17S??$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?0?*???!V?.??P@)?J?4??18?/??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??H?}}?!???]?@)??H?}}?1???]?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;?O??nr?!?W???@);?O??nr?1?W???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapˡE?????!?d???i@@)?~j?t?h?1???"7@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9޵??e???IJn?t?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Dio?????Dio?????!Dio?????      ??!       "      ??!       *      ??!       2	?=yX?9@?=yX?9@!?=yX?9@:      ??!       B      ??!       J	?W?2???W?2??!?W?2??R      ??!       Z	?W?2???W?2??!?W?2??b      ??!       JCPU_ONLYY޵??e???b qJn?t?X@