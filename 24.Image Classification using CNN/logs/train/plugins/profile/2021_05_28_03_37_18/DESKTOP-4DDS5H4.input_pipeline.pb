	󫺌Eo]@󫺌Eo]@!󫺌Eo]@	?=^ゎ;!@?=^ゎ;!@!?=^ゎ;!@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$󫺌Eo]@?|a2?6@A詺?U@Y_楲孞$@*	烫烫题聾2F
Iterator::Model歸湤#9$@!?$5??覺@)aTR'?)$@1dd圀罄X@:Preprocessing2U
Iterator::Model::ParallelMapV2俿F旜??!?<拉???)俿F旜??1?<拉???:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatT悭浤 ??!?I涜逃?)傥鱏悭??1囆嘸区??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?#裹圁??!?贳)??)虬Pk歸??1詅8T儁??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceS?!巙q??!酹沜権??)S?!巙q??1酹沜権??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip霶?呺??!?痬? ??)?5?;N褋?1顗佽镞??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;逴崡nr?!_k
?	ˇ?);逴崡nr?1_k
?	ˇ?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2娈%鋬??!掁q5Y灰?){瓽醶d?1踆}樀$??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t19.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?=^ゎ;!@IF8t+傌V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?|a2?6@?|a2?6@!?|a2?6@      ??!       "      ??!       *      ??!       2	詺?U@詺?U@!詺?U@:      ??!       B      ??!       J	_楲孞$@_楲孞$@!_楲孞$@R      ??!       Z	_楲孞$@_楲孞$@!_楲孞$@b      ??!       JCPU_ONLYY?=^ゎ;!@b qF8t+傌V@