#pragma once

#include "volume_interpolation.h"

#include <cuda_fp16.h>
#include <filesystem>
#include <cuda_runtime.h>

BEGIN_RENDERER_NAMESPACE

struct InputParametrization
{
    //iff true, the time is passed as additional input after the position
    bool hasTime = false;
    bool hasDirection = false;
    int numFourierFeatures = 0;
    bool useDirectionInFourierFeatures = false;
    typedef std::vector<half> FourierMatrix_t;
    /**
     * The B*3 matrix of fourier features 'F' where B is the number of fourier features.
     * The input position p is multiplied with this matrix and concatenated as follows
     * to produce the final output:
     *
     * p = [px, py, pz] the input position
     * out = [p; cos(2*pi*F*p); sin(2*pi*F*p)]
     * where sin and cos is applied point-wise.
     *
     * Furthermore, if useDirectionInFourierFeatures==true
     * (implies hasDirection==true), then the input is a six-dimensional vector
     * [px,py,pz, dx,dy,dz] including the direction.
     */
    FourierMatrix_t fourierMatrix;

    [[nodiscard]] int channelsOut() const;

    [[nodiscard]] bool valid() const;

private:
    static const int VERSION;
public:
    [[nodiscard]] static std::shared_ptr<InputParametrization> load(std::istream& in);
    void save(std::ostream& out) const;
    /**
     * Sets the fourier matrix from the pytorch tensor
     * \param t the fourier matrix tensor
     * \param premultiplied true if the tensor is already scaled with 2*pi,
     *   false if not. If false, the scaling is done here.
     */
    void setFourierMatrixFromTensor(const torch::Tensor& t, bool premultiplied);
    void disableFourierFeatures();
};
typedef std::shared_ptr<InputParametrization> InputParametrization_ptr;
typedef std::shared_ptr<const InputParametrization> InputParametrization_cptr;

struct OutputParametrization
{
    enum OutputMode
    {
        DENSITY,
        DENSITY_DIRECT,
        RGBO,
        RGBO_DIRECT,
        _NUM_OUTPUT_MODES_
    };
    static const std::string OutputModeNames[_NUM_OUTPUT_MODES_]; //match the python def.
    static const int OutputModeNumChannels[_NUM_OUTPUT_MODES_];
    OutputMode outputMode = DENSITY;
    [[nodiscard]] static OutputMode OutputModeFromString(const std::string& s);

    [[nodiscard]] int channelsIn() const;
    [[nodiscard]] int channelsOut() const;

private:
    static const int VERSION;
public:
    [[nodiscard]] static std::shared_ptr<OutputParametrization> load(std::istream& in);
    void save(std::ostream& out) const;
};
typedef std::shared_ptr<OutputParametrization> OutputParametrization_ptr;
typedef std::shared_ptr<const OutputParametrization> OutputParametrization_cptr;

/**
 * An inner layer of the scene network.
 * The weights are stored as matrix 'weights(channel_out, channel_in)',
 * i.e. the weight matrix has num-channel-out rows and num-channel-in columns.
 */
struct Layer
{
    using weights_t = std::vector<half>;
    using bias_t = std::vector<half>;

    int channelsIn;
    int channelsOut;
    weights_t weights; //output fastest, input slowest
    bias_t bias;
    enum Activation
    {
        ReLU,
        Sine,
        Snake,
        SnakeAlt,
        Sigmoid,
        None, //for testing
        _NUM_ACTIVATIONS_
    };
    static const std::string ActivationNames[_NUM_ACTIVATIONS_];
    [[nodiscard]] static Activation ActivationFromString(const std::string& s);
    Activation activation;
    float activationParameter;

    Layer(int channelsIn, int channelsOut, 
        const weights_t& weights, const bias_t& bias, Activation activation, float activationParameter)
        : channelsIn(channelsIn),
        channelsOut(channelsOut),
        weights(weights),
        bias(bias),
        activation(activation),
        activationParameter(activationParameter)
    {
        TORCH_CHECK(channelsIn*channelsOut == weights.size(), "dimensions and weights not compatible");
        TORCH_CHECK(channelsOut == bias.size(), "dimensions and bias not compatible");
    }

    [[nodiscard]] bool valid() const;

private:
    static const int VERSION;
public:
    [[nodiscard]] static std::shared_ptr<Layer> load(std::istream& in);
    void save(std::ostream& out) const;
};
typedef std::shared_ptr<Layer> Layer_ptr;
typedef std::shared_ptr<const Layer> Layer_cptr;

/**
 * Container for the hidden layers of a single SRN.
 * They are further grouped in InnerNetworks below.
 */
class InnerNetwork
{
    std::vector<Layer_ptr> hidden_;

public:
    InnerNetwork() = default;

    /**
     * \brief Adds a hidden layer.
     * The input parametrization / input channels for the first layer is assumed to be:
     *  [0:2]: position,
     *  [2:4]: direction, if available
     *  [2:] or [4:]: fourier features
     * \param layer the layer to add
     */
    void addLayer(InputParametrization_cptr input, Layer_ptr layer);
    //Adds a hidden layer without preprocessing
    void addLayerRaw(Layer_ptr layer) { hidden_.push_back(layer); }

    void addLayerFromTorch(InputParametrization_ptr input, const torch::Tensor& weights, const torch::Tensor& bias,
        Layer::Activation activation, float activationParameter = 1);

    [[nodiscard]] int numLayers() const { return hidden_.size(); }
    [[nodiscard]] Layer_cptr getHidden(int index) const;
    [[nodiscard]] Layer_ptr getHidden(int index);
    [[nodiscard]] const std::vector<Layer_ptr>& getLayers() const { return hidden_; }

    //Returns the input channels of the first layer
    [[nodiscard]] int numInputChannels() const;
    //Returns the output channels of the last layer
    [[nodiscard]] int numOutputChannels() const;

    /**
     * Checks that the layers are compatible,
     * i.e. the output channels of layer i match the input channels of layer i+1.
     */
    [[nodiscard]] bool valid() const;

private:
    static const int VERSION;
public:
    [[nodiscard]] static std::shared_ptr<InnerNetwork> load(std::istream& in);
    void save(std::ostream& out) const;
};
typedef std::shared_ptr<InnerNetwork> InnerNetwork_ptr;
typedef std::shared_ptr<const InnerNetwork> InnerNetwork_cptr;

/**
 * Storage for 'm' hidden networks, each of 'n' layers.
 * Each hidden network shares the same architecture, just different weights.
 * Furthermore, to predict multiple parameters, the output of the
 * last layer can be a multiple 'p' of the channels required by
 * the output parametrization
 */
class InnerNetworks
{
    std::vector<InnerNetwork_ptr> networks_;
    int numOutputParameters_ = 1;

public:
    InnerNetworks() = default;

    [[nodiscard]] int numInnerNetworks() const { return networks_.size(); }
    /**
     * Adds a new inner network and appends it to the list
     */
    [[nodiscard]] InnerNetwork_ptr addInnerNetwork();
    //Returns the inner network with index in {0, numInnerNetworks()-1}
    [[nodiscard]] InnerNetwork_ptr getInnerNetwork(int index);
    [[nodiscard]] InnerNetwork_cptr getInnerNetwork(int index) const;

    /**
     * To support multiple output parameters,
     * the networks can output a multiple of the channels required by the output
     * parametrization. This property stores that multiply
     */
    [[nodiscard]] int numOutputParameters() const { return numOutputParameters_; }
    void setNumOutputParameters(int p);

    /**
     * Checks that all inner networks have the same shape
     * and that the output matches the output parametrization
     * up to a multiplication of numOutputParameters()
     */
    [[nodiscard]] bool valid(int inputChannels, OutputParametrization_cptr out) const;

private:
    static const int VERSION;
public:
    [[nodiscard]] static std::shared_ptr<InnerNetworks> load(std::istream& in);
    void save(std::ostream& out) const;
};
typedef std::shared_ptr<InnerNetworks> InnerNetworks_ptr;
typedef std::shared_ptr<const InnerNetworks> InnerNetworks_cptr;

struct LatentGridTimeAndEnsemble;

/**
 * Description for the volumetric latent grid.
 * This contains the storage and quantization mappings for a single timestep+ensemble.
 */
class LatentGrid
{
public:
    /**
     * Describes how the latent grid is encoded in the 3D texture
     */
    enum Encoding
    {
        //Saved as 32-bit float without any extra parameter mapping
        FLOAT,
        //Saved as 8-bit unsigned byte that is expanded to a float in [0,1] (variable 'x')
        //by the texture hardware and then linearly mapped to the requested range (value(x))
        //via the given offset and scale:
        //value(x) = offset + x * scale
        BYTE_LINEAR,
        //Saved as 8-bit unsigned byte that is expanded to a float in [0,1]
        //by the texture hardware and then mapped to the Gaussian
        //with the given mean and variance.
        //value(x) = mean + sigma * \Theta^-1(x)
        BYTE_GAUSSIAN,
    };

    static constexpr float ENCODING_GAUSSIAN_EPSILON = 1e-4f;

    using grid_t = std::vector<char>;
    using grid_coeff_t = std::vector<float>;

    /**
     * The encoding type
     */
    Encoding encoding = FLOAT;

    /**
     * The grid, a vector of type float or char.
     *
     * The grid is a 5D object of size
     * (gridChannelsHigh, gridSizeZ, gridSizeY, gridSizeX, gridChannelsLow)
     * where gridChannelsLow is fastest, gridChannelsHigh is slowest.
     * gridChannelsLow is in {0,1,2,3}, gridChannelsHigh in {0,1,...,gridChannels/4-1}.
     *
     * This is for easy copying to the GPU:
     * On the GPU, the grid is saved as a collection of 3D textures of four channels.
     * In that format, the four channels are fastest, followed by x, y, and z (slowest).
     * This is repeated gridChannels/4 times.
     */
    grid_t grid;

    //The number of channels. Must be a multiple of 16 for the TensorCore implementation
    int gridChannels = 0;
    int gridSizeZ = 0;
    int gridSizeY = 0;
    int gridSizeX = 0;

    [[nodiscard]] size_t bytesPerEntry() const { return encoding == FLOAT ? sizeof(float) : 1; }

    /**
     * For encoding==BYTE_LINEAR, specifies the offset per grid channel
     * For encoding==BYTE_GAUSSIAN, specifies the mean per grid channel
     */
    grid_coeff_t gridOffsetOrMean;

    /**
     * For encoding==BYTE_LINEAR, specifies the scale per grid channel
     * For encoding==BYTE_GAUSSIAN, specifies the standard deviation per grid channel
     */
    grid_coeff_t gridScaleOrStd;

private:
    static const int VERSION;
    [[nodiscard]] int idx(int cHigh, int z, int y, int x, int cLow) const
    {
        return cLow + 4 * (x + gridSizeX * (y + gridSizeY * (z + gridSizeZ * cHigh)));
    }
    struct GPUArray
    {
        cudaArray_t array;
        cudaTextureObject_t texture;
        GPUArray(int sizeX, int sizeY, int sizeZ, bool isFloat, const char* data);
        ~GPUArray();
    };
    std::vector<std::shared_ptr<GPUArray>> gpuResources_;

public:
    LatentGrid() = default;
    /**
     * Initializes the latent grid from the given PyTorch tensor of shape (1,C,Z,Y,X).
     * This method automatically computes the scale and offsets for the
     * selected encoding method.
     */
    LatentGrid(const torch::Tensor& t, Encoding encoding);
private:
    void initEncodingFloat(const torch::Tensor& t);
    void initEncodingByteLinear(const torch::Tensor& t);
    void initEncodingByteGaussian(const torch::Tensor& t);
    static double LastEncodingError;
    friend struct LatentGridTimeAndEnsemble;

public:
    [[nodiscard]] bool isValid() const;

    void clearGPUResources();

    void copyGridToGPU(bool skipIfAlreadyInitialized);

    /**
     * Returns the texture for the given channel index.
     * Note that always four channels are grouped together in a texture.
     * Hence, 0<=index<gridChannels/4
     */
    [[nodiscard]] cudaTextureObject_t getTexture(int index) const;

    /**
     * Returns the offset (encoding=BYTE_LINEAR) or mean (encoding=BYTE_GAUSSIAN)
     * for the given channel index.
     * Note that always four channels are grouped together in a texture and processed together.
     * Hence, 0<=index<gridChannels/4
     */
    [[nodiscard]] float4 getOffsetOrMean(int index) const;

    /**
     * Returns the offset (encoding=BYTE_LINEAR) or mean (encoding=BYTE_GAUSSIAN)
     * for the given channel index.
     * Note that always four channels are grouped together in a texture and processed together.
     * Hence, 0<=index<gridChannels/4
     */
    [[nodiscard]] float4 getScaleOrStd(int index) const;

    [[nodiscard]] static std::shared_ptr<LatentGrid> load(std::istream& in);
    void save(std::ostream& out) const;
};
typedef std::shared_ptr<LatentGrid> LatentGrid_ptr;
typedef std::shared_ptr<const LatentGrid> LatentGrid_cptr;

/**
 * Container for time- and ensemble dependent Latent Grids.
 *
 * Ranges for the time keyframes are specified as:
 *  timeMin, timeNum, timeStep
 * The minimal timestep is 'timeMin', the maximal timestep is 'timeMin+(timeNum-1)*timeStep'.
 * Allowed time values are 't in [timeMin, timeMin+(timeNum-1)*timeStep]',
 * keyframes are where '(t-timeMin)%timeStep==0', linear interpolation in between.
 *
 * Ensembles are specified via ensembleMin, ensembleNum.
 */
struct LatentGridTimeAndEnsemble
{
    int timeMin = 0;
    int timeNum = 0;
    int timeStep = 1;
    //The time-dependent latent grids of length 'timeNum'.
    std::vector<LatentGrid_ptr> timeGrids;

    int ensembleMin = 0;
    int ensembleNum = 0;
    std::vector<LatentGrid_ptr> ensembleGrids;

private:
    static const int VERSION;

public:
    LatentGridTimeAndEnsemble() = default;

    LatentGridTimeAndEnsemble(int time_min, int time_num, int time_step, int ensemble_min, int ensemble_num)
        : timeMin(time_min),
          timeNum(time_num),
          timeStep(time_step),
          timeGrids(time_num),
          ensembleMin(ensemble_min),
          ensembleNum(ensemble_num),
          ensembleGrids(ensemble_num)
    {}

    [[nodiscard]] bool hasTimeGrids() const { return timeNum > 0; }
    [[nodiscard]] bool hasEnsembleGrids() const { return ensembleNum > 0; }

    /**
     * Returns the maximal inclusive timestep
     */
    [[nodiscard]] int timeMaxInclusive() const { return timeMin + (timeNum-1)*timeStep; }

    [[nodiscard]] int ensembleMaxInclusive() const { return ensembleMin + ensembleNum - 1; }

    /**
     * Computes the interpolation index into the list of time grids
     * based on the given absolute time in [timeMin, timeMin+timeNum-1]
     */
    [[nodiscard]] float interpolateTime(float time) const
    {
        float v = (time - timeMin) / timeStep;
        return clamp(v, 0.f, static_cast<float>(timeNum-1));
    }

    [[nodiscard]] int interpolateEnsemble(int ensemble) const
    {
        return clamp(ensemble - ensembleMin, 0, ensembleNum - 1);
    }

    [[nodiscard]] LatentGrid_ptr getTimeGrid(int index)
    {
        TORCH_CHECK(index >= 0 && index < timeNum, "Index out of bounds");
        return timeGrids[index];
    }
    [[nodiscard]] LatentGrid_cptr getTimeGrid(int index) const
    {
        TORCH_CHECK(index >= 0 && index < timeNum, "Index out of bounds");
        return timeGrids[index];
    }

    [[nodiscard]] LatentGrid_ptr getEnsembleGrid(int index)
    {
        TORCH_CHECK(index >= 0 && index < ensembleNum, "Index out of bounds");
        return ensembleGrids[index];
    }
    [[nodiscard]] LatentGrid_cptr getEnsembleGrid(int index) const
    {
        TORCH_CHECK(index >= 0 && index < ensembleNum, "Index out of bounds");
        return ensembleGrids[index];
    }

    /**
     * Sets the grid at the given index from the pytorch tensor.
     * See LatentGrid::LatentGrid(torch::Tensor, Encoding).
     * Returns the average encoding error
     */
    double setTimeGridFromTorch(int index, const torch::Tensor& t, LatentGrid::Encoding encoding);

    /**
     * Sets the grid at the given index from the pytorch tensor.
     * See LatentGrid::LatentGrid(torch::Tensor, Encoding).
     * Returns the average encoding error
     */
    double setEnsembleGridFromTorch(int index, const torch::Tensor& t, LatentGrid::Encoding encoding);

    /**
     * Checks if this latent grid collection is valid.
     * This means, all time grids have the same channel count.
     * And all ensemble grids have the same channel count.
     * The encoding over all grids must also match.
     */
    [[nodiscard]] bool isValid() const;

    void clearGPUResources();

    //for the ui, example resolution.
    //In theory, the grids could have different resolutions.
    //This just returns the first resolution
    [[nodiscard]] int getResolution() const;

    //For code generation:

    [[nodiscard]] LatentGrid::Encoding getCommonEncoding() const;
    [[nodiscard]] int getTimeChannels() const;
    [[nodiscard]] int getEnsembleChannels() const;
    [[nodiscard]] int getTotalChannels() const { return getTimeChannels() + getEnsembleChannels(); }

public:
    [[nodiscard]] static std::shared_ptr<LatentGridTimeAndEnsemble> load(std::istream& in);
    void save(std::ostream& out) const;
};
typedef std::shared_ptr<LatentGridTimeAndEnsemble> LatentGridTimeAndEnsemble_ptr;
typedef std::shared_ptr<const LatentGridTimeAndEnsemble> LatentGridTimeAndEnsemble_cptr;

class SceneNetwork;
typedef std::shared_ptr<SceneNetwork> SceneNetwork_ptr;

/**
 * Specification of the scene network.
 * You must set the input parametrization first before adding the hidden layers.
 *
 * <b>Important:</b>
 * It is assumed that the network is fully configured before the first render.
 * Before rendering it for the first time, GPU resources and settings are automatically
 * allocated and cached.
 * To clear those GPU resources (in case you do want to edit it afterwards), call
 * \ref clearGPUResources()
 */
class SceneNetwork
{
    //Network specs
    InputParametrization_ptr input_;
    OutputParametrization_ptr output_;
    //std::vector<Layer_ptr> hidden_;
    InnerNetworks_ptr inner_;
    LatentGridTimeAndEnsemble_ptr latentGrid_;
    float3 boxMin_;
    float3 boxSize_;

    //Selected variation
    float currentTime_ = 0;
    int currentEnsemble_ = 0;
    int currentNetwork_ = 0;
    int currentParameter_ = 0;

    //GPU cache
    mutable std::vector<char> cacheConstantMemory_;
    mutable std::string cacheDefines_;

public:
    SceneNetwork();

    [[nodiscard]] InputParametrization_cptr input() const { return input_; }
    [[nodiscard]] InputParametrization_ptr input() { return input_; }

    [[nodiscard]] OutputParametrization_cptr output() const { return output_; }
    [[nodiscard]] OutputParametrization_ptr output() { return output_; }

    [[nodiscard]] InnerNetworks_cptr networks() const { return inner_; }
    [[nodiscard]] InnerNetworks_ptr networks() { return inner_; }

    [[nodiscard]] LatentGridTimeAndEnsemble_cptr latentGrid() const { return latentGrid_; }
    [[nodiscard]] LatentGridTimeAndEnsemble_ptr latentGrid() { return latentGrid_; }
    void setLatentGrid(LatentGridTimeAndEnsemble_ptr g) { latentGrid_ = g; }

    [[nodiscard]] float3 boxMin() const { return boxMin_; }
    void setBoxMin(const float3& boxMin) { boxMin_ = boxMin; }

    [[nodiscard]] float3 boxSize() const { return boxSize_; }
    void setBoxSize(const float3& boxSize) { boxSize_ = boxSize; }

    /**
     * Sets the time and ensemble in the latent grid to be used for rendering.
     * See \ref LatentSpaceTimeAndEnsemble for the allowed bounds
     */
    void setTimeAndEnsemble(float time, int ensemble);
    [[nodiscard]] float currentTime() const { return currentTime_; }
    [[nodiscard]] int currentEnsemble() const { return currentEnsemble_; }

    /**
     * Multiple hidden networks can be specified that share
     * the latent grid, input and output encoding.
     * This is another way how ensemble members could be encoded besides the latent grid.
     * Bounds: {0, networks->numInnerNetworks()-1}
     */
    void setCurrentNetwork(int i);
    [[nodiscard]] int currentNetwork() const { return currentNetwork_; }

    /**
     * The networks can predict multiple parameters at once by outputting
     * a multiple of the channels required for the output parametrization in
     * the last layer.
     * Here, it is selected, which of those parameters is used.
     * Bounds: {0, networks->numOutputParameters()-1}
     */
    void setCurrentParameter(int p);
    [[nodiscard]] int currentParameter() const { return currentParameter_; }

    /**
     * Checks if this network is valid, all layers fit together.
     * \return false if invalid
     */
    bool valid() const;

    /**
     * Computes the maximal number of warps that are possible
     * if this network is evaluated using tensor cores.
     * The number of warps is limited by the available shared memory.
     * This method returns a negative number if the network is too big.
     */
    int computeMaxWarps(bool onlySharedMemory) const;
    /**
     * computes the number of parameters
     */
    int numParameters() const;
    /**
     * Frees any GPU resources. They are recreated on demand
     */
    void clearGPUResources();

private:
    static const int VERSION;
public:
    static SceneNetwork_ptr load(std::istream & in);
    void save(std::ostream & out) const;
    
    /**
     * The return type of the evaluation function
     */
    std::string codeReturnType() const;

public:
    [[nodiscard]] std::string getDefines(const IKernelModule::GlobalSettings& s, 
        int numWarps, bool firstAndLastInSharedMemory) const;
    [[nodiscard]] std::vector<std::string> getIncludeFileNames(const IKernelModule::GlobalSettings& s) const;
    [[nodiscard]] std::string getConstantDeclarationName(const IKernelModule::GlobalSettings& s) const;
    [[nodiscard]] std::string getPerThreadType(const IKernelModule::GlobalSettings& s) const;
    void fillConstantMemory(const IKernelModule::GlobalSettings& s, CUdeviceptr ptr, CUstream stream);
};

/**
 * \brief Volume interpolation using a neural network
 * (Scene Representation Network) via Tensor Cores.
 */
class VolumeInterpolationNetwork : public IVolumeInterpolation
{
    static constexpr int MAX_BLOCK_SIZE = 256; //probably too conservative
    struct LoadedNetwork
    {
        SceneNetwork_ptr network;
        int numWarpsSharedOnly;
        int numWarpsMixed;
        std::string filename;
        std::string humanname;
    };
    std::vector<LoadedNetwork> networks_;
    int selectedNetwork_;

    bool onlySharedMemory_; //true -> bias and fourier features are in shared memory, not constant

    bool hasTimesteps_ = false;
    bool hasEnsembles_ = false;
    bool hasMultiNetworks_ = false;
    bool hasMultiParameters_ = false;
    int currentEnsemble_ = 0;
    float currentTimestep_ = 0;
    int currentMultiNetwork_ = 0;
    int currentMultiParameter_ = 0;

    //rendering intermediates
    mutable int currentNumWarps_;
    mutable bool currentOnlyShared_;
    mutable int currentTargetBlockSize_;

public:
    VolumeInterpolationNetwork();

private:
    void addNetwork(SceneNetwork_ptr network, const std::string& filename);
    void selectNetwork(int index);
    void loadNetworkDialog();
public:
    /**
     * UI: loads a network from the file and adds it to the list.
     */
    void loadNetwork(const std::string& filename);
    /**
     * Python: replaces all loaded networks by this network and activate it
     */
    void setNetwork(SceneNetwork_ptr network);

    bool onlySharedMemory() const { return onlySharedMemory_; }
    void setOnlySharedMemory(bool b) { onlySharedMemory_ = b; }

    void setTimeAndEnsemble(float time, int ensemble);
    void setInnerNetwork(int index);
    void setOutputParameter(int index);

    [[nodiscard]] std::string getName() const override;
    [[nodiscard]] bool drawUI(UIStorage_t& storage) override;
    void load(const nlohmann::json& json, const ILoadingContext* fetcher) override;
    void save(nlohmann::json& json, const ISavingContext* context) const override;
    void prepareRendering(GlobalSettings& s) const override;
    [[nodiscard]] GlobalSettings::VolumeOutput outputType() const override;
    [[nodiscard]] std::optional<int> getBatches(const GlobalSettings& s) const override;
    [[nodiscard]] std::string getDefines(const GlobalSettings& s) const override;
    [[nodiscard]] std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
    [[nodiscard]] std::string getConstantDeclarationName(const GlobalSettings& s) const override;
    [[nodiscard]] std::string getPerThreadType(const GlobalSettings& s) const override;
    void fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream) override;
protected:
    void registerPybindModule(pybind11::module& m) override;
};

END_RENDERER_NAMESPACE
