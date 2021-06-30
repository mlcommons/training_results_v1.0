// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "dropout_with_switch.cpp"

class DropoutWithTrainingSwitchOpX : public popart::popx::ElementWiseBinaryOpx
{
public:
  DropoutWithTrainingSwitchOpX(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::ElementWiseBinaryOpx(op, devicex) {
    verifyOp<DropoutWithTrainingSwitchOp>(op, CustomOperators::DropoutWithTrainingSwitch);
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto &op          = getOp<DropoutWithTrainingSwitchOp>();
    const poplar::Tensor &isTraining = getInTensor(op.getTrainingSwitchInIndex());
    const poplar::Tensor &activation = getInTensor(op.getInIndex());
	  const poplar::Tensor &refTensor = get(op.getReferenceTensorId());
    double dropoutProbability = 1. - static_cast<double>(op.getRatio());
    double scale = 1. / (1. - static_cast<double>(op.getRatio()));
	  

    prog.add(poplar::program::AssumeEqualAcrossReplicas(isTraining[0]));

    // Perform dropout (will clone input tensor)
    auto dropout = poprand::dropout(graph(),
                                    &getInTensor(op.getSeedInIndex()),
                                    0u,
                                    activation,
                                    refTensor,
                                    dropoutProbability,
                                    scale,
                                    prog,
                                    debugContext("dropout"));
    
    // During evaluation this op should should return the input
    // so just overwrite the dropout tensor with the input
    auto evaluation_sequence = poplar::program::Sequence();
    evaluation_sequence.add(poplar::program::Copy(activation, dropout));
    prog.add(poplar::program::If(isTraining[0], poplar::program::Sequence(), evaluation_sequence));
  
    setOutTensor(op.getOutIndex(), dropout);
	  }
  };

static popart::popx::OpxCreator<DropoutWithTrainingSwitchOpX>
  DropoutWithTrainingSwitchOpxCreator(CustomOperators::DropoutWithTrainingSwitch);
