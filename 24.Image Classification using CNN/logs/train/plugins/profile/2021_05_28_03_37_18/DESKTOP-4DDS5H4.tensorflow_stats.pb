"?K
BHostIDLE"IDLE1?????{?@A?????{?@a??l7L-??i??l7L-???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1???̬$?@9???̬$?@A???̬$?@I???̬$?@a?'?H???i????-????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1????,??@9????,??@A????,??@I????,??@a-??tD??i??w>?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1????l?@9????l?@A????l?@I????l?@a?tg-纽?i9?$$n???Unknown?
oHost_FusedMatMul"sequential/dense/Relu(13333???@93333???@A3333???@I3333???@aӫF?t???i?;s?????Unknown
dHostDataset"Iterator::Model(1????̿?@9????̿?@Affff???@Iffff???@a?F\?n??i??݈?????Unknown
^HostGatherV2"GatherV2(13333s|?@93333s|?@A3333s|?@I3333s|?@a?s[l;??i?S?Nf???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1    ??@9    ??@A    ??@I    ??@a???ߢ?i:}?6eA???Unknown
{	HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1????L??@9????L??@A????L??@I????L??@a!? m???i,???j???Unknown
}
HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1fffffI?@9fffffI?@AfffffI?@IfffffI?@aZy?)???i??)?2???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1??????@9??????@A??????@I??????@a?(?
0Ӕ?i?po?-????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1?????at@9?????at@A?????at@I?????at@a????T?i??wt<????Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1fffff?c@9fffff?c@Afffff?c@Ifffff?c@a??Ð?dC?i??[?????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1??????X@9??????X@A??????X@I??????X@a?[?r??8?i0;?~&????Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      V@9      V@A      V@I      V@a??e???5?i?ǜQ?????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1fffff?T@9fffff?T@Afffff?T@Ifffff?T@a??`???4?i4?fq????Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1fffff?R@9fffff?R@Afffff?R@Ifffff?R@a???HT?2?i?MQ?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1?????L>@9?????L>@A?????L>@I?????L>@a<??????i?\???????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1??????;@9??????;@A??????;@I??????;@aCe?|p?iQB??????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ??@9     ??@A      ;@I      ;@a?;????i#9>f????Unknown
iHostWriteSummary"WriteSummary(1??????8@9??????8@A??????8@I??????8@a??Z?G?i?cz(????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1?????L4@9?????L4@A?????L4@I?????L4@a[? ?	?i̌??????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1?????3@9?????3@A?????3@I?????3@a?@A????i?q?_????Unknown
eHost
LogicalAnd"
LogicalAnd(1??????,@9??????,@A??????,@I??????,@aێ??(:?i8z?????Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1?????L;@9?????L;@A??????+@I??????+@aCe?|p?i?*3:>????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1??????+@9??????+@A??????+@I??????+@a?:%=?iÿ/0?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????*@9??????*@A??????*@I??????*@aF?d?]s
?iWݦ?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1ffffff(@9ffffff(@Affffff(@Iffffff(@a?t????i픍Qu????Unknown
|HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1??????%@9??????%@A??????%@I??????%@a?b{??Q?i?*???????Unknown
ZHostArgMax"ArgMax(1ffffff%@9ffffff%@Affffff%@Iffffff%@a????i?b?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1??????$@9??????$@A??????$@I??????$@a??00?T?i?# gp????Unknown
g HostStridedSlice"strided_slice(1ffffff#@9ffffff#@Affffff#@Iffffff#@a?0q#?%?i|????????Unknown
l!HostIteratorGetNext"IteratorGetNext(1??????!@9??????!@A??????!@I??????!@a??Q?^?i??y????Unknown
x"HostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?Q@9     ?Q@Affffff!@Iffffff!@a?HܸU,?i3?s*G????Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      !@9      !@A      !@I      !@a˳?	E? ?i???G?????Unknown
?$HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1ffffff @9ffffff @Affffff @Iffffff @a?ԑ?/ ?iA8?????Unknown
`%HostGatherV2"
GatherV2_1(1333333 @9333333 @A333333 @I333333 @ah9XG??>i????
????Unknown
?&HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1??????@9??????@A??????@I??????@a??c?%0?>i{?[I????Unknown
?'HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@an?????>iDs???????Unknown
?(HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1??????@9??????@A??????@I??????@aF?d?]s?>iR??????Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1??????@9??????@A??????@I??????@a?2??+D?>iXw?m?????Unknown
V*HostSum"Sum_2(1333333@9333333@A333333@I333333@a????>i͎?+????Unknown
?+HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@a?JP?J?>i?.??J????Unknown
v,HostCast"$sparse_categorical_crossentropy/Cast(1??????@9??????@A??????@I??????@aS!;????>id??w????Unknown
?-HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1??????@9??????@A??????@I??????@aS!;????>i??kĤ????Unknown
?.HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a?'<????>iRvIL?????Unknown
u/HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?'<????>i??&??????Unknown
t0HostAssignAddVariableOp"AssignAddVariableOp(1??????@9??????@A??????@I??????@a??Q?^?>im??????Unknown
?1HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@afi|????>ifz\?/????Unknown
?2HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1333333@9333333@A333333@I333333@a?Ovn??>i??ʓJ????Unknown
X3HostCast"Cast_4(1??????	@9??????	@A??????	@I??????	@a?2??+D?>i[???c????Unknown
X4HostEqual"Equal(1??????@9??????@A??????@I??????@aM	?\
z?>i+R|????Unknown
b5HostDivNoNan"div_no_nan_1(1333333@9333333@A333333@I333333@a?%????>iQ??7?????Unknown
?6HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_2(1ffffff@9ffffff@Affffff@Iffffff@a??PC??>i??nS?????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff@9ffffff@Affffff@Iffffff@a??PC??>i?/o?????Unknown
?8HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1??????@9??????@A??????@I??????@a?b{??Q?>in???????Unknown
V9HostCast"Cast(1      @9      @A      @I      @a??)B??>i???}?????Unknown
?:HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1??????=@9??????=@A      @I      @a??)B??>ii;?????Unknown
?;HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1333333@9333333@A333333@I333333@a*??? ??>i5?.????Unknown
?<HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1333333@9333333@A333333@I333333@a*??? ??>i`!"????Unknown
X=HostCast"Cast_3(1ffffff@9ffffff@Affffff@Iffffff@a??&n?(?>i/o_J4????Unknown
T>HostMul"Mul(1ffffff@9ffffff@Affffff@Iffffff@a??&n?(?>iV?^sF????Unknown
`?HostDivNoNan"
div_no_nan(1??????@9??????@A??????@I??????@a??Q?^?>i??<?W????Unknown
w@HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?N?6??>iOB؜g????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??an,??? ?>i!9R?v????Unknown
?BHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1????????9????????A????????I????????a@??1?l?>iҪӄ????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_3(1333333??9333333??A333333??I333333??a?Ovn??>iF???????Unknown
?DHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1333333??9333333??A333333??I333333??a?Ovn??>inH??????Unknown
aEHostIdentity"Identity(1????????9????????A????????I????????a?2??+D?>i?%/N?????Unknown?
sFHostReadVariableOp"SGD/Cast/ReadVariableOp(1????????9????????A????????I????????a?2??+D?>iE??????Unknown
wGHostReadVariableOp"div_no_nan/ReadVariableOp_1(1????????9????????A????????I????????a?2??+D?>ig?Z??????Unknown
?HHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1????????9????????A????????I????????a?2??+D?>i??p4?????Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a???????>i7=e?????Unknown
?JHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a???????>i??Y??????Unknown
yKHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a??PC??>i\?,??????Unknown
?LHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a??PC??>i     ???Unknown*?K
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1???̬$?@9???̬$?@A???̬$?@I???̬$?@a???Ň[??i???Ň[???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1????,??@9????,??@A????,??@I????,??@a???Xu???i?A4r???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1????l?@9????l?@A????l?@I????l?@agwn??M??i?????!???Unknown?
oHost_FusedMatMul"sequential/dense/Relu(13333???@93333???@A3333???@I3333???@av;?Ԯ?i
vi?*???Unknown
dHostDataset"Iterator::Model(1????̿?@9????̿?@Affff???@Iffff???@aW$??lw??iO?^?????Unknown
^HostGatherV2"GatherV2(13333s|?@93333s|?@A3333s|?@I3333s|?@a?r???&??ixt?????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1    ??@9    ??@A    ??@I    ??@aˠWM8???i??I^????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1????L??@9????L??@A????L??@I????L??@a?'??iu:@?????Unknown
}	HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1fffffI?@9fffffI?@AfffffI?@IfffffI?@a=???)???i??(??????Unknown
q
Host_FusedMatMul"sequential/dense_1/Relu(1??????@9??????@A??????@I??????@a??P?R??iH:)A#????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1?????at@9?????at@A?????at@I?????at@ac?0[8?_?i??V??????Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1fffff?c@9fffff?c@Afffff?c@Ifffff?c@a??爐gN?i?y??????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1??????X@9??????X@A??????X@I??????X@a?w"0?9C?i;E,P????Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      V@9      V@A      V@I      V@a???36A?i??y?????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1fffff?T@9fffff?T@Afffff?T@Ifffff?T@a??}?Y+@?ia-KP?????Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1fffff?R@9fffff?R@Afffff?R@Ifffff?R@a#Uub{>=?i|?D????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1?????L>@9?????L>@A?????L>@I?????L>@a{9?q'?i???0?????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1??????;@9??????;@A??????;@I??????;@a%b??%?i???O????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ??@9     ??@A      ;@I      ;@a??
?|?$?i???a????Unknown
iHostWriteSummary"WriteSummary(1??????8@9??????8@A??????8@I??????8@a???'#?iH??	?????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1?????L4@9?????L4@A?????L4@I?????L4@ar[?C	i?i??GR?????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1?????3@9?????3@A?????3@I?????3@a?Â9???i????y????Unknown
eHost
LogicalAnd"
LogicalAnd(1??????,@9??????,@A??????,@I??????,@a?d@` ?i?
??*????Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1?????L;@9?????L;@A??????+@I??????+@a%b???i(]??????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1??????+@9??????+@A??????+@I??????+@atО?QZ?i?줁????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????*@9??????*@A??????*@I??????*@a9??H???i???'????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1ffffff(@9ffffff(@Affffff(@Iffffff(@a?[3>???i?I??????Unknown
|HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1??????%@9??????%@A??????%@I??????%@a9U?\???i??38D????Unknown
ZHostArgMax"ArgMax(1ffffff%@9ffffff%@Affffff%@Iffffff%@a??N?`??i?w;??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1??????$@9??????$@A??????$@I??????$@a_?g????i???*H????Unknown
gHostStridedSlice"strided_slice(1ffffff#@9ffffff#@Affffff#@Iffffff#@a??|??i???<?????Unknown
l HostIteratorGetNext"IteratorGetNext(1??????!@9??????!@A??????!@I??????!@a%???;?il4+-????Unknown
x!HostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?Q@9     ?Q@Affffff!@Iffffff!@a???Q?
?i??Gܘ????Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      !@9      !@A      !@I      !@aK?ug?M
?i?(?????Unknown
?#HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1ffffff @9ffffff @Affffff @Iffffff @arnM?4`	?i????g????Unknown
`$HostGatherV2"
GatherV2_1(1333333 @9333333 @A333333 @I333333 @a????	?i?ވ??????Unknown
?%HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1??????@9??????@A??????@I??????@a?"%]?r?iPS??-????Unknown
?&HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a???R5??i?????????Unknown
?'HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1??????@9??????@A??????@I??????@a9??H???i????????Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_2(1??????@9??????@A??????@I??????@a`?[?5??i!??'*????Unknown
V)HostSum"Sum_2(1333333@9333333@A333333@I333333@a?8N??~?iZ?#x????Unknown
?*HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@aM~?B?i??9,?????Unknown
v+HostCast"$sparse_categorical_crossentropy/Cast(1??????@9??????@A??????@I??????@a??᧣?i?Gٺ????Unknown
?,HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1??????@9??????@A??????@I??????@a??᧣?i??xIN????Unknown
?-HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a_㸚???>ip??????Unknown
u.HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a_㸚???>i?:g??????Unknown
t/HostAssignAddVariableOp"AssignAddVariableOp(1??????@9??????@A??????@I??????@a%???;?>i?{(?????Unknown
?0HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@a?Kh????>i?4?%(????Unknown
?1HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1333333@9333333@A333333@I333333@a?a??>i?s?;R????Unknown
X2HostCast"Cast_4(1??????	@9??????	@A??????	@I??????	@a`?[?5??>iu?e?y????Unknown
X3HostEqual"Equal(1??????@9??????@A??????@I??????@a%?@?/?>i?$?7?????Unknown
b4HostDivNoNan"div_no_nan_1(1333333@9333333@A333333@I333333@a?????>i???????Unknown
?5HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_2(1ffffff@9ffffff@Affffff@Iffffff@at2?
oT?>i?????????Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff@9ffffff@Affffff@Iffffff@at2?
oT?>i??ko	????Unknown
?7HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1??????@9??????@A??????@I??????@a9U?\???>ix|f?*????Unknown
V8HostCast"Cast(1      @9      @A      @I      @a?5?4??>i?}??I????Unknown
?9HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1??????=@9??????=@A      @I      @a?5?4??>i?~οh????Unknown
?:HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1333333@9333333@A333333@I333333@a{	?P??>i?#u?????Unknown
?;HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1333333@9333333@A333333@I333333@a{	?P??>i?o*?????Unknown
X<HostCast"Cast_3(1ffffff@9ffffff@Affffff@Iffffff@a???Hmx?>i?ݢ?????Unknown
T=HostMul"Mul(1ffffff@9ffffff@Affffff@Iffffff@a???Hmx?>i?ZJ?????Unknown
`>HostDivNoNan"
div_no_nan(1??????@9??????@A??????@I??????@a%???;?>iNG?V?????Unknown
w?HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a8?24???>i?{?????Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a????߄?>i~Sw?(????Unknown
?AHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1????????9????????A????????I????????aL?{?G?>iE?s?>????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1333333??9333333??A333333??I333333??a?a??>i????S????Unknown
?CHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1333333??9333333??A333333??I333333??a?a??>ig??h????Unknown
aDHostIdentity"Identity(1????????9????????A????????I????????a`?[?5??>i????|????Unknown?
sEHostReadVariableOp"SGD/Cast/ReadVariableOp(1????????9????????A????????I????????a`?[?5??>i???????Unknown
wFHostReadVariableOp"div_no_nan/ReadVariableOp_1(1????????9????????A????????I????????a`?[?5??>i{XGf?????Unknown
?GHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1????????9????????A????????I????????a`?[?5??>i?}4?????Unknown
uHHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??%gR??>i?????????Unknown
?IHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??%gR??>i#?!W?????Unknown
yJHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??at2?
oT?>i????????Unknown
?KHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??at2?
oT?>i     ???Unknown2CPU