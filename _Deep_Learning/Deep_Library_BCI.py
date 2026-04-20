import tensorflow as tf

from tensorflow.keras import layers, models, losses, optimizers, metrics

class MultiTaskModel(tf.keras.Model):
    def __init__(
        self,
        input_shape = (128,1),
        latent_dim=16,
        conv_filters=(16, 64, 256, 128),
        kernel_size=3,
        activation_fn=layers.LeakyReLU,
        use_decoder=True,
        use_classifier=True,
        latent_activation =  None, # use tanh if you want to compare with handextracted features ???
        use_GRU = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_shape_ = input_shape
        self.latent_dim = latent_dim
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.activation_fn = activation_fn
        self.latent_activation = latent_activation

        # Build components
        if use_GRU:
            self.encoder = self._build_encoder_gru()
        else:
            self.encoder = self._build_encoder()
        self.decoder = self._build_decoder() if use_decoder else None
        self.classifier = self._build_classifier() if use_classifier else None

    # --------------------------------------------------
    # ENCODER
    # --------------------------------------------------
    def _build_encoder(self):
        inputs = layers.Input(shape=self.input_shape_, name="encoder_input")

        x = inputs

        for n_layer in range(len(self.conv_filters)):
            filters = self.conv_filters[n_layer]
            x = layers.Conv1D(filters, self.kernel_size, padding="same")(x)
            x = layers.BatchNormalization()(x)      # TODO vedi qui      
            x = self.activation_fn()(x)
            x = layers.MaxPooling1D()(x)                        # downsample by 2 n_filters times (4-> shape is 128/16 = 8)
            if n_layer < len(self.conv_filters) -1 :
                x = layers.Dropout(0.15)(x)

        x = layers.Flatten()(x) # 8 * 128 = 1024

        # x = layers.Dense(self.latent_dim*2)(x) # TODO yes or no?? (Attenttion to batch norm here)
        #x = self.activation_fn(x)
        latent = layers.Dense(self.latent_dim, name="latent", activation=self.latent_activation)(x)

        return models.Model(inputs, latent, name="encoder")

    def _build_encoder_gru(self):
        inputs = layers.Input(shape=self.input_shape_, name="encoder_input")

        x = inputs

        for n_layer in range(len(self.conv_filters)):
            filters = self.conv_filters[n_layer]
            x = layers.Conv1D(filters, self.kernel_size, padding="same")(x)
            x = layers.BatchNormalization()(x)      # TODO vedi qui      
            x = self.activation_fn()(x)
            x = layers.MaxPooling1D()(x)                        # downsample by 2 n_filters times (4-> shape is 128/16 = 8)
            if n_layer < len(self.conv_filters) -1 :
                x = layers.Dropout(0.15)(x)
            if n_layer == 1:
                x = layers.Bidirectional( layers.GRU(64, return_sequences=True), merge_mode="ave" ) (x)

        x = layers.Flatten()(x)
        # x = layers.Dense(self.latent_dim*2)(x) # TODO yes or no?? (Attenttion to batch norm here)
        #x = self.activation_fn(x)
        latent = layers.Dense(self.latent_dim, name="latent", activation=self.latent_activation)(x)

        return models.Model(inputs, latent, name="encoder")

    # --------------------------------------------------
    # DECODER
    # --------------------------------------------------
    def _build_decoder(self):
        if self.input_shape_[0] % (2 ** len(self.conv_filters)) != 0:
            raise ValueError("Input length must be divisible by total downsampling factor")
        
        latent_inputs = layers.Input(shape=(self.latent_dim,), name="decoder_input")

        downsample_factor = 2 ** len(self.conv_filters)
        h = self.input_shape_[0] // downsample_factor
        c = self.conv_filters[-1]
        print(h,c)
        x = layers.Dense(h * c)(latent_inputs)
        x = self.activation_fn()(x)
        x = layers.Reshape((h, c))(x)

        for filters in reversed(self.conv_filters):
            x = layers.UpSampling1D()(x)
            x = layers.Conv1D(filters, self.kernel_size, padding="same")(x)
            # x = layers.Conv1DTranspose(filters, self.kernel_size, strides=2, padding="same")(x)
            x = self.activation_fn()(x)

        outputs = layers.Conv1D(
            self.input_shape_[-1],
            kernel_size=3,
            padding="same",
            activation="linear",   # TODO attention to the normalization of data
            name="reconstruction"
        )(x)

        return models.Model(latent_inputs, outputs, name="decoder")

    # --------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------
    def _build_classifier(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,), name="classifier_input")

        outputs = layers.Dense(
            1,
            activation="sigmoid",
            name="classification"
        )(latent_inputs)

        return models.Model(latent_inputs, outputs, name="classifier")

    # --------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------
    def call(self, inputs, training=False):

        latent = self.encoder(inputs, training=training)

        outputs = {}

        if self.decoder is not None:
            outputs["reconstruction"] = self.decoder(latent, training=training)

        if self.classifier is not None:
            outputs["classification"] = self.classifier(latent, training=training)

        # Return single tensor if only one head exists
        if len(outputs) == 1:
            return next(iter(outputs.values()))

        return outputs

    # --------------------------------------------------
    # COMPILE FOR DIFFERENT CASES
    # --------------------------------------------------
    def compile_cases(self, optimizer, loss_reconstruction=None, loss_classification=None,
                      loss_weights=None, metric_reconstruction=None, metric_classification=None):

        
        if loss_weights is None:
            loss_weights = {"reconstruction": 1.0, "classification": 1.0}
        if metric_reconstruction is None:
            metric_reconstruction = ["mae"]
        if metric_classification is None:
            metric_classification = ["accuracy"]
        
        # Classifier-only
        if self.decoder is None and self.classifier is not None:
            self.compile(optimizer=optimizer, loss=loss_classification, metrics=metric_classification)

        # Autoencoder-only
        elif self.decoder is not None and self.classifier is None:
            self.compile(optimizer=optimizer, loss=loss_reconstruction, metrics=metric_reconstruction)

        # Multi-output
        elif self.decoder is not None and self.classifier is not None:
            loss_fns = {"reconstruction": loss_reconstruction,
            "classification": loss_classification}
            metrics_dict= {"reconstruction": metric_reconstruction,
                           "classification": metric_classification}

            self.compile(optimizer=optimizer, loss=loss_fns, metrics=metrics_dict, loss_weights=loss_weights)

        else:
            raise RuntimeError("Model has neither decoder nor classifier.")
        

    # --------------------------------------------------
    # DEFINE FIT CASES
    # --------------------------------------------------

    def fit_cases(self, x_train, x_val, y_train=None, y_val=None, epochs=100, batch_size=64, **kwargs):
        
        # Classifier-only
        if self.decoder is None and self.classifier is not None:
            storia = self.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, **kwargs)
            return storia

        # Autoencoder-only
        elif self.decoder is not None and self.classifier is None:
            storia = self.fit(x_train, x_train, validation_data=(x_val,x_val), epochs=epochs, batch_size=batch_size, **kwargs)
            return storia

        # Multi-output
        elif self.decoder is not None and self.classifier is not None:
            storia = self.fit(x_train, {"reconstruction": x_train, "classification": y_train}, validation_data=(x_val, {"reconstruction": x_val, "classification": y_val}), epochs=epochs, batch_size=batch_size, **kwargs)
            return storia

        else:
            raise RuntimeError("Model has neither decoder nor classifier.")  


class MultiTaskModel_PowerSpectra(tf.keras.Model):
    """
    The Runnig class of the models.

    """
    def __init__(
        self,
        input_shape = (128,1),
        output_shape = None,
        latent_dim=16,
        conv_filters=(16, 64, 256, 128),
        kernel_size=3,
        activation_fn=layers.LeakyReLU,
        use_decoder=True,
        use_classifier=True,
        latent_activation =  None, # use tanh if you want to compare with handextracted features ???
        use_GRU = False,
        build_convolutional_decoder = False,
        convolutional_latent = True,
        EEG_concat_MEG = False, # not so explicit, hard to understand. used when I have multimodality EEG and MEG
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_shape_ = input_shape
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.activation_fn = activation_fn
        self.latent_activation = latent_activation
        self.conv_decoder = build_convolutional_decoder
        self.EEG_concat_MEG = EEG_concat_MEG
        self.convolutional_latent = convolutional_latent

        # Build components
        if use_GRU:
            self.encoder = self._build_encoder_gru()
        else:
            self.encoder = self._build_encoder()
        self.decoder = self._build_decoder() if use_decoder else None
        self.classifier = self._build_classifier() if use_classifier else None

    # --------------------------------------------------
    # ENCODER
    # --------------------------------------------------
    def _build_encoder(self):
        inputs = layers.Input(shape=self.input_shape_, name="encoder_input")

        x = inputs

        for n_layer in range(len(self.conv_filters)):
            filters = self.conv_filters[n_layer]
            x = layers.Conv1D(filters, self.kernel_size, padding="same")(x)
            # x = layers.BatchNormalization()(x)            
            x = self.activation_fn()(x)
            x = layers.AveragePooling1D(pool_size=2)(x)       # downsample by 2 n_filters times (4-> shape is 128/16 = 8)
            if n_layer < len(self.conv_filters) -1 :
                x = layers.Dropout(0.15)(x)


        if self.convolutional_latent:
            # Add an hand-crafted locally connected layer as last layer.
            # It is useful to not have weights sharing, because each latent dimension 
            # can learn to encode a specific mode, so it should not share weights with the other dimensions.
            # in this way it is a little bit more a Dense layer, but preserve the convolutional structure
            # And i can use a convolutional structure for the second taraining phase

            def apply_mask_for_locally_connected(t):
                # Here i am trying to build a locally connected layer
                # Keras does not support anymore the locally connected, so I implement mine
                length = tf.shape(t)[1]
                mask = tf.eye(length)
                mask = tf.reshape(mask, (1, length, length))
                return t * mask
            
            def check_length(t):
                length = tf.shape(t)[1]
                tf.debugging.assert_equal(length, self.latent_dim, 
                        message=f"Actual latent dim ({length}) is not the desired one ({self.latent_dim})!")
                return t
            
            x = layers.Conv1D(filters//2, self.kernel_size, padding="same")(x)
            x = self.activation_fn()(x)
    
            x = layers.Lambda(check_length)(x)
            
            # Build my locally connected layer
            x = layers.Conv1D(self.latent_dim, 1, padding="same", activation="linear")(x)
            print("shape_print number 1", x.shape)
            x = layers.Lambda(apply_mask_for_locally_connected, name="masked_for_locally_connected")(x)
            print("shape_print number 2", x.shape)
            latent = layers.Lambda(lambda t: tf.reduce_sum(t, axis=-1, keepdims=True),
                                name="latent")(x)

            print("shape_print number 3", latent.shape)


        else:
            x = layers.Flatten()(x) # 16 * 128 = 2048
            latent = layers.Dense(self.latent_dim, name="latent", activation=self.latent_activation)(x)

        return models.Model(inputs, latent, name="encoder")
    

    def _build_encoder_gru(self):
        # used if i want to use a gru layer 
        # (all the tests say it is not useful)

        inputs = layers.Input(shape=self.input_shape_, name="encoder_input")

        x = inputs

        for n_layer in range(len(self.conv_filters)):
            filters = self.conv_filters[n_layer]
            x = layers.Conv1D(filters, self.kernel_size, padding="same")(x)
            x = layers.BatchNormalization()(x)      # TODO vedi qui      
            x = self.activation_fn()(x)
            if n_layer < len(self.conv_filters) -1 :
                x = layers.MaxPooling1D()(x)                        # downsample by 2 n_filters times (4-> shape is 128/16 = 8)
                x = layers.Dropout(0.15)(x)

        x = layers.Bidirectional( layers.GRU(16) ) (x)

        # x = layers.Dense(self.latent_dim*2)(x) # TODO yes or no?? (Attenttion to batch norm here)
        #x = self.activation_fn(x)
        latent = layers.Dense(self.latent_dim, name="latent", activation=self.latent_activation)(x)

        return models.Model(inputs, latent, name="encoder")

    # --------------------------------------------------
    # DECODER
    # --------------------------------------------------
    def _build_decoder(self):

        # CONVOLUTIONAL - IF NEEDED I ouput a convolutional structure as dcoder 
        # (as output a Conv1D, not a Dense layer)
        input_shape = (self.latent_dim,1) if self.convolutional_latent else (self.latent_dim,)

        if self.conv_decoder:

            latent_inputs = layers.Input(shape=input_shape, name="decoder_input")
            if self.convolutional_latent:
                x = layers.Reshape((self.latent_dim,))(latent_inputs)
            else:
                x = latent_inputs
            x = layers.Dense(32*64)(x)
            x = layers.Reshape((32,64))(x)

            for num,filters in enumerate(reversed(self.conv_filters[0:3])):
                if num != 2:
                    x = layers.Conv1D(filters, self.kernel_size+1)(x)
                    x = self.activation_fn()(x)
                else:
                    x = layers.Conv1D(1+self.EEG_concat_MEG, self.kernel_size)(x)

            x = layers.Activation("linear")(x)
            
            if self.EEG_concat_MEG:
                outputs = x
            else:
                outputs = layers.Flatten()(x)

        else:
            latent_inputs = layers.Input(shape=input_shape, name="decoder_input")
            if self.convolutional_latent:
                x = layers.Reshape((self.latent_dim,))(latent_inputs)
            else:
                x = latent_inputs
            x = layers.Dense(16*128)(x)
            x = layers.Reshape((16, 128))(x)

            for num,filters in enumerate(reversed(self.conv_filters[0:2])):
                if num == 0:
                    x = layers.UpSampling1D()(x)
                x = layers.Conv1D(filters, self.kernel_size, padding="same")(x)
                x = self.activation_fn()(x)
            x = layers.Flatten()(x)
            x = layers.Dense(self.output_shape)(x)
            outputs = layers.Activation("linear", name="reconstruction")(x)

        return models.Model(latent_inputs, outputs, name="decoder")

    # --------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------
    def _build_classifier(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,), name="classifier_input")

        outputs = layers.Dense(
            1,
            activation="sigmoid",
            name="classification"
        )(latent_inputs)

        return models.Model(latent_inputs, outputs, name="classifier")

    # --------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------
    def call(self, inputs, training=False):

        latent = self.encoder(inputs, training=training)

        outputs = {}

        if self.decoder is not None:
            outputs["reconstruction"] = self.decoder(latent, training=training)

        if self.classifier is not None:
            outputs["classification"] = self.classifier(latent, training=training)

        # Return single tensor if only one head exists
        if len(outputs) == 1:
            return next(iter(outputs.values()))

        return outputs

    # --------------------------------------------------
    # COMPILE FOR DIFFERENT CASES
    # --------------------------------------------------
    def compile_cases(self, optimizer, loss_reconstruction=None, loss_classification=None,
                      loss_weights=None, metric_reconstruction=None, metric_classification=None):

        
        if loss_weights is None:
            loss_weights = {"reconstruction": 1.0, "classification": 1.0}
        if metric_reconstruction is None:
            metric_reconstruction = ["mae"]
        if metric_classification is None:
            metric_classification = ["accuracy"]
        
        # Classifier-only
        if self.decoder is None and self.classifier is not None:
            self.compile(optimizer=optimizer, loss=loss_classification, metrics=metric_classification)

        # Autoencoder-only
        elif self.decoder is not None and self.classifier is None:
            self.compile(optimizer=optimizer, loss=loss_reconstruction, metrics=metric_reconstruction)

        # Multi-output
        elif self.decoder is not None and self.classifier is not None:
            loss_fns = {"reconstruction": loss_reconstruction,
            "classification": loss_classification}
            metrics_dict= {"reconstruction": metric_reconstruction,
                           "classification": metric_classification}

            self.compile(optimizer=optimizer, loss=loss_fns, metrics=metrics_dict, loss_weights=loss_weights)

        else:
            raise RuntimeError("Model has neither decoder nor classifier.")
        

    # --------------------------------------------------
    # DEFINE FIT CASES
    # --------------------------------------------------

    def fit_cases(self, x_train, power_train, x_val, power_val, y_train=None, y_val=None, epochs=100, batch_size=64, **kwargs):
        
        # Classifier-only
        if self.decoder is None and self.classifier is not None:
            storia = self.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, **kwargs)
            return storia

        # Autoencoder-only
        elif self.decoder is not None and self.classifier is None:
            storia = self.fit(x_train, power_train, validation_data=(x_val,power_val), epochs=epochs, batch_size=batch_size, **kwargs)
            return storia

        # Multi-output
        elif self.decoder is not None and self.classifier is not None:
            storia = self.fit(x_train, {"reconstruction": power_train, "classification": y_train}, validation_data=(x_val, {"reconstruction":power_val, "classification": y_val}), epochs=epochs, batch_size=batch_size, **kwargs)
            return storia

        else:
            raise RuntimeError("Model has neither decoder nor classifier.")  


class MultiTaskModel_Multimodal(MultiTaskModel):
    """
    Used to take as  input      EEG (or MEG) 
    and reconstruct the other   MEG (or EEG)

    not used so much, i tested that this is a little failure
    """
    # --------------------------------------------------
    # RE - DEFINE FIT CASES
    # --------------------------------------------------

    def fit_cases(self, x_train, x_train_output, x_val, x_val_output, y_train=None, y_val=None, epochs=100, batch_size=64, **kwargs):
        
        # Classifier-only
        if self.decoder is None and self.classifier is not None:
            storia = self.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, **kwargs)
            return storia

        # Autoencoder-only
        elif self.decoder is not None and self.classifier is None:
            storia = self.fit(x_train, x_train_output, validation_data=(x_val,x_val_output), epochs=epochs, batch_size=batch_size, **kwargs)
            return storia

        # Multi-output
        elif self.decoder is not None and self.classifier is not None:
            storia = self.fit(x_train, {"reconstruction": x_train_output, "classification": y_train}, validation_data=(x_val, {"reconstruction": x_val_output, "classification": y_val}), epochs=epochs, batch_size=batch_size, **kwargs)
            return storia

        else:
            raise RuntimeError("Model has neither decoder nor classifier.")  

