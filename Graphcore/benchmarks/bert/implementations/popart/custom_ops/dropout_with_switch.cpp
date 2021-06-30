// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op.hpp>
#include <popart/shapeinference.hpp>
#include <popart/names.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/op/dropout.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/devicex.hpp>
#include <poprand/RandomGen.hpp>
#include <random>

using namespace popart;
using namespace popart::popx;

namespace CustomOperators {
  const popart::OperatorIdentifier DropoutWithTrainingSwitch = {"ai.graphcore", "DropoutWithTrainingSwitch", 1};
} // namespace CustomOperators

class DropoutWithTrainingSwitchOp : public popart::DropoutOp {
public:
    using popart::DropoutOp::DropoutOp;
    InIndex getTrainingSwitchInIndex() const { return 1; }
    InIndex getSeedInIndex() const override { return 2; }
    InIndex getGradInIndex() const { return 0; }

    std::unique_ptr<Op> clone() const override {
     return std::make_unique<DropoutWithTrainingSwitchOp>(*this);
    }
    void setup() {outInfo(getOutIndex()) = inInfo(getInIndex());}

    
  const std::vector<GradInOutMapper> &gradInputInfo() const override {
	  static const std::vector<GradInOutMapper> inInfo = {
	      {getGradInIndex(), DropoutWithTrainingSwitchOp::getOutIndex(), GradOpInType::GradOut},
          {getTrainingSwitchInIndex(), DropoutWithTrainingSwitchOp::getTrainingSwitchInIndex(), GradOpInType::In},
	      {getSeedInIndex(), DropoutWithTrainingSwitchOp::getSeedInIndex(), GradOpInType::In}};
	  return inInfo;
	}
	
	const std::map<int, int> &gradOutToNonGradIn() const override {
	  static const std::map<int, int> outInfo = {
	      {getOutIndex(), getInIndex()}};
	  return outInfo;
	}

  std::vector<std::unique_ptr<Op>> getGradOps() override {
	  std::vector<std::unique_ptr<Op>> upops;
	  upops.emplace_back(std::make_unique<DropoutWithTrainingSwitchOp>(*this));
	  return upops;
	}
};

static OpDefinition::DataTypes T  = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes B = {DataType::UINT32};

static popart::OpDefinition DropoutWithTrainingSwitchOpDef({
                  OpDefinition::Inputs({{"data", T}, {"mode", B}}),
                  OpDefinition::Outputs({{"output", T}}),
                  OpDefinition::Attributes({{"ratio", {"*"}}})});

static popart::OpCreator<DropoutWithTrainingSwitchOp> DropoutWithTrainingSwitchOpCreator(
    popart::OpDefinitions({{CustomOperators::DropoutWithTrainingSwitch, DropoutWithTrainingSwitchOpDef}}),
    [](const popart::OpCreatorInfo &oci) -> std::unique_ptr<popart::Op> {
      float ratio = oci.attributes.getAttribute<Attributes::Float>("ratio");
      return std::unique_ptr<DropoutWithTrainingSwitchOp>(
          new DropoutWithTrainingSwitchOp(oci.opid, ratio, oci.settings));
    },
    true);

static popart::RegisterShapeInferenceFunction DropoutWithTrainingShapeInfer(
          CustomOperators::DropoutWithTrainingSwitch,
          [](ShapeInferenceContext &ctx) {
              ctx.outInfo(0) = ctx.inInfo(0);
          });
