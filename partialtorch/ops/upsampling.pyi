from typing import overload, Sequence, Optional, Union

from partialtorch.types import _float, _bool, _symint, MaskedPair, _MaskedPairOrTensor


# upsample_nearest
@overload
def upsample_nearest1d(self: _MaskedPairOrTensor,
                       output_size: Optional[Union[_symint, Sequence[_symint]]],
                       scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def upsample_nearest1d(self: _MaskedPairOrTensor,
                       output_size: Union[_symint, Sequence[_symint]],
                       scales: Optional[_float] = None) -> MaskedPair: ...


@overload
def _upsample_nearest_exact1d(self: _MaskedPairOrTensor,
                              output_size: Optional[Union[_symint, Sequence[_symint]]],
                              scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def _upsample_nearest_exact1d(self: _MaskedPairOrTensor,
                              output_size: Union[_symint, Sequence[_symint]],
                              scales: Optional[_float] = None) -> MaskedPair: ...


@overload
def upsample_nearest2d(self: _MaskedPairOrTensor,
                       output_size: Optional[Union[_symint, Sequence[_symint]]],
                       scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def upsample_nearest2d(self: _MaskedPairOrTensor,
                       output_size: Union[_symint, Sequence[_symint]],
                       scales_h: Optional[_float] = None,
                       scales_w: Optional[_float] = None) -> MaskedPair: ...


@overload
def _upsample_nearest_exact2d(self: _MaskedPairOrTensor,
                              output_size: Optional[Union[_symint, Sequence[_symint]]],
                              scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def _upsample_nearest_exact2d(self: _MaskedPairOrTensor,
                              output_size: Union[_symint, Sequence[_symint]],
                              scales_h: Optional[_float] = None,
                              scales_w: Optional[_float] = None) -> MaskedPair: ...


@overload
def upsample_nearest3d(self: _MaskedPairOrTensor,
                       output_size: Optional[Union[_symint, Sequence[_symint]]],
                       scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def upsample_nearest3d(self: _MaskedPairOrTensor,
                       output_size: Union[_symint, Sequence[_symint]],
                       scales_d: Optional[_float] = None,
                       scales_h: Optional[_float] = None,
                       scales_w: Optional[_float] = None) -> MaskedPair: ...


@overload
def _upsample_nearest_exact3d(self: _MaskedPairOrTensor,
                              output_size: Optional[Union[_symint, Sequence[_symint]]],
                              scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def _upsample_nearest_exact3d(self: _MaskedPairOrTensor,
                              output_size: Union[_symint, Sequence[_symint]],
                              scales_d: Optional[_float] = None,
                              scales_h: Optional[_float] = None,
                              scales_w: Optional[_float] = None) -> MaskedPair: ...


# upsample_lerp
@overload
def upsample_linear1d(self: _MaskedPairOrTensor,
                      output_size: Optional[Union[_symint, Sequence[_symint]]],
                      align_corners: _bool,
                      scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def upsample_linear1d(self: _MaskedPairOrTensor,
                      output_size: Union[_symint, Sequence[_symint]],
                      align_corners: _bool,
                      scales: Optional[_float] = None) -> MaskedPair: ...


@overload
def upsample_bilinear2d(self: _MaskedPairOrTensor,
                        output_size: Optional[Union[_symint, Sequence[_symint]]],
                        align_corners: _bool,
                        scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def upsample_bilinear2d(self: _MaskedPairOrTensor,
                        output_size: Union[_symint, Sequence[_symint]],
                        align_corners: _bool,
                        scales_h: Optional[_float] = None,
                        scales_w: Optional[_float] = None) -> MaskedPair: ...


@overload
def _upsample_bilinear2d_aa(self: _MaskedPairOrTensor,
                            output_size: Optional[Union[_symint, Sequence[_symint]]],
                            align_corners: _bool,
                            scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def _upsample_bilinear2d_aa(self: _MaskedPairOrTensor,
                            output_size: Union[_symint, Sequence[_symint]],
                            align_corners: _bool,
                            scales_h: Optional[_float] = None,
                            scales_w: Optional[_float] = None) -> MaskedPair: ...


@overload
def upsample_trilinear3d(self: _MaskedPairOrTensor,
                         output_size: Optional[Union[_symint, Sequence[_symint]]],
                         align_corners: _bool,
                         scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def upsample_trilinear3d(self: _MaskedPairOrTensor,
                         output_size: Union[_symint, Sequence[_symint]],
                         align_corners: _bool,
                         scales_d: Optional[_float] = None,
                         scales_h: Optional[_float] = None,
                         scales_w: Optional[_float] = None) -> MaskedPair: ...


# partial_upsample_lerp
@overload
def partial_upsample_linear1d(self: _MaskedPairOrTensor,
                              output_size: Optional[Union[_symint, Sequence[_symint]]],
                              align_corners: _bool,
                              scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def partial_upsample_linear1d(self: _MaskedPairOrTensor,
                              output_size: Union[_symint, Sequence[_symint]],
                              align_corners: _bool,
                              scales: Optional[_float] = None) -> MaskedPair: ...


@overload
def partial_upsample_bilinear2d(self: _MaskedPairOrTensor,
                                output_size: Optional[Union[_symint, Sequence[_symint]]],
                                align_corners: _bool,
                                scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def partial_upsample_bilinear2d(self: _MaskedPairOrTensor,
                                output_size: Union[_symint, Sequence[_symint]],
                                align_corners: _bool,
                                scales_h: Optional[_float] = None,
                                scales_w: Optional[_float] = None) -> MaskedPair: ...


@overload
def _partial_upsample_bilinear2d_aa(self: _MaskedPairOrTensor,
                                    output_size: Optional[Union[_symint, Sequence[_symint]]],
                                    align_corners: _bool,
                                    scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def _partial_upsample_bilinear2d_aa(self: _MaskedPairOrTensor,
                                    output_size: Union[_symint, Sequence[_symint]],
                                    align_corners: _bool,
                                    scales_h: Optional[_float] = None,
                                    scales_w: Optional[_float] = None) -> MaskedPair: ...


@overload
def partial_upsample_trilinear3d(self: _MaskedPairOrTensor,
                                 output_size: Optional[Union[_symint, Sequence[_symint]]],
                                 align_corners: _bool,
                                 scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def partial_upsample_trilinear3d(self: _MaskedPairOrTensor,
                                 output_size: Union[_symint, Sequence[_symint]],
                                 align_corners: _bool,
                                 scales_d: Optional[_float] = None,
                                 scales_h: Optional[_float] = None,
                                 scales_w: Optional[_float] = None) -> MaskedPair: ...


# upsample_bicubic
@overload
def upsample_bicubic2d(self: _MaskedPairOrTensor,
                       output_size: Optional[Union[_symint, Sequence[_symint]]],
                       align_corners: _bool,
                       scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def upsample_bicubic2d(self: _MaskedPairOrTensor,
                       output_size: Union[_symint, Sequence[_symint]],
                       align_corners: _bool,
                       scales_h: Optional[_float] = None,
                       scales_w: Optional[_float] = None) -> MaskedPair: ...


@overload
def _upsample_bicubic2d_aa(self: _MaskedPairOrTensor,
                           output_size: Optional[Union[_symint, Sequence[_symint]]],
                           align_corners: _bool,
                           scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def _upsample_bicubic2d_aa(self: _MaskedPairOrTensor,
                           output_size: Union[_symint, Sequence[_symint]],
                           align_corners: _bool,
                           scales_h: Optional[_float] = None,
                           scales_w: Optional[_float] = None) -> MaskedPair: ...


# partial_upsample_bicubic
@overload
def partial_upsample_bicubic2d(self: _MaskedPairOrTensor,
                               output_size: Optional[Union[_symint, Sequence[_symint]]],
                               align_corners: _bool,
                               scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def partial_upsample_bicubic2d(self: _MaskedPairOrTensor,
                               output_size: Union[_symint, Sequence[_symint]],
                               align_corners: _bool,
                               scales_h: Optional[_float] = None,
                               scales_w: Optional[_float] = None) -> MaskedPair: ...


@overload
def _partial_upsample_bicubic2d_aa(self: _MaskedPairOrTensor,
                                   output_size: Optional[Union[_symint, Sequence[_symint]]],
                                   align_corners: _bool,
                                   scale_factors: Optional[_float]) -> MaskedPair: ...


@overload
def _partial_upsample_bicubic2d_aa(self: _MaskedPairOrTensor,
                                   output_size: Union[_symint, Sequence[_symint]],
                                   align_corners: _bool,
                                   scales_h: Optional[_float] = None,
                                   scales_w: Optional[_float] = None) -> MaskedPair: ...
