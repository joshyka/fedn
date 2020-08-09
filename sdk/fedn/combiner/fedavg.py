import time
import json
import queue
import tempfile
import os
import tensorflow as tf
from fedn.utils.helpers import KerasSequentialHelper
from fedn.utils.mongo import connect_to_mongodb
import fedn.proto.alliance_pb2 as alliance
from fedn.combiner.server import CombinerClient

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# TODO rename to Combiner
class FEDAVGCombiner(CombinerClient):
    """ A Local SGD / Federated Averaging (FedAvg) combiner. """

    def __init__(self, address, port, id, role, storage):

        super().__init__(address, port, id, role)

        self.storage = storage
        self.id = id
        self.model_id = None

        self.data = {}
        # self.config = config
        # TODO  refactor since we are now getting config on RUN cmd.
        self.db = connect_to_mongodb()
        self.coll = self.db['orchestrators']

        self.config = {}
        # TODO: Use MongoDB
        self.validations = {}

        # TODO: make choice of helper configurable
        self.helper = KerasSequentialHelper()
        # Queue for model updates to be processed.
        self.model_updates = queue.Queue()

    def __set_model(self, model_id):
        self.model_id = model_id
        try:
            result = self.coll.update_one({'id': self.id}, {'$set': {'model_id': self.model_id}})
        except Exception as e:
            self.report_status("FEDAVG: FAILED TO UPDATED MONGODB RECORD {}".format(e))

    def get_model_id(self):
        # Check if there is another model available in the db
        # TODO check timestamped model id if a newer is available in db.. (local is set due to crash?)
        # data = self.coll.find_one({'id':self.id})
        # mid = data['model_id']

        return self.model_id

    def report_status(self, msg, log_level=alliance.Status.INFO, type=None, request=None, flush=True):
        print("ORCHESTRATOR({}):{} {}".format(self.id, log_level, msg), flush=flush)

    def receive_model_candidate(self, model_id):
        """ Callback when a new model version has been trained. 
            We simply put the model_id on a queue to be processed later. """
        try:
            self.report_status("ORCHESTRATOR: callback received model {}".format(model_id),
                               log_level=alliance.Status.INFO)
            # TODO - here would be a place to do some additional validation of the model contribution. 
            self.model_updates.put(model_id)
        except Exception as e:
            self.report_status("ORCHESTRATOR: Failed to receive candidate model! {}".format(e),
                               log_level=alliance.Status.WARNING)
            print("Failed to receive candidate model!")
            pass

    def receive_validation(self, validation):
        """ Callback for a validation request """

        # TODO: Track this in a DB
        model_id = validation.model_id
        data = json.loads(validation.data)
        try:
            self.validations[model_id].append(data)
        except KeyError:
            self.validations[model_id] = [data]

        self.report_status("ORCHESTRATOR: callback processed validation {}".format(validation.model_id),
                           log_level=alliance.Status.INFO)

    def combine_models(self, nr_expected_models=None, timeout=120):
        """ Compute an iterative/running average of models arriving to the combiner. """

        round_time = 0.0
        print("ORCHESTRATOR: combining model updates...")

        # First model in the update round
        try:
            model_id = self.model_updates.get(timeout=timeout)
            print("combining ", model_id)
            model_str = self.storage.get_model(model_id)
            model = self.helper.load_model(model_str)
            nr_processed_models = 1
            self.model_updates.task_done()
        except queue.Empty as e:
            self.report_status("ORCHESTRATOR: training round timed out.", log_level=alliance.Status.WARNING)
            # print("ORCHESTRATOR: Round timed out.")
            return None

        while nr_processed_models < nr_expected_models:
            try:
                model_id = self.model_updates.get(block=False)
                self.report_status("Received model update with id {}".format(model_id))

                model_next = self.helper.load_model(self.storage.get_model(model_id))
                self.helper.increment_average(model, model_next, nr_processed_models)

                nr_processed_models += 1
                self.model_updates.task_done()
            except Exception as e:
                self.report_status("ORCHESTRATOR failed!!!!: {0}".format(e))
                time.sleep(1.0)
                round_time += 1.0

            if round_time >= timeout:
                self.report_status("ORCHESTRATOR: training round timed out.", log_level=alliance.Status.WARNING)
                print("ORCHESTRATOR: Round timed out.")
                return None

        self.report_status("ORCHESTRATOR: Training round completed, combined {} models.".format(nr_processed_models),
                           log_level=alliance.Status.INFO)
        print("DONE, combined {} models".format(nr_processed_models))
        return model

    def __assign_clients(self, n):
        # Obtain a list of clients to talk to
        # TODO: If we want global sampling without replacement the server needs to assign clients
        active_trainers = self.get_active_trainers()
        import random
        # import numpy
        self.trainers = random.sample(active_trainers, n)
        # self.trainers = numpy.random.choice(active_trainers,n,replace=False)
        # TODO: In the general case, validators could be other clients as well
        self.validators = self.trainers
        # TODO, update MongoDB entry
        # try:
        #    result = self.coll.update_one({'id':self.id},{'$set':{'model_id':self.model_id}})
        # except Exception as e:
        #    print("FEDAVG: FAILED TO UPDATED MONGODB RECORD {}".format(e), flush=True)

    def __training_round(self):

        # We flush the queue at a beginning of a round (no stragglers allowed)
        # TODO: Support other ways to handle stragglers. 
        with self.model_updates.mutex:
            self.model_updates.queue.clear()

        self.report_status("ORCHESTRATOR: Initiating training round, participating members: {}".format(self.trainers))
        self.request_model_update(self.model_id, clients=self.trainers)

        # Apply combiner
        model = self.combine_models(nr_expected_models=len(self.trainers), timeout=self.config['round_timeout'])
        return model

    def __validation_round(self):
        self.request_model_validation(self.model_id, from_clients=self.validators)

    def run(self, config):
        # rounds=1, active_clients=2):
        # This (2 mins) is the max we wait in a training round before moving on.
        self.config['round_timeout'] = 120
        # Parameters that are passed on to trainers to control local optimization settings.
        self.config['nr_local_epochs'] = 1
        self.config['local_batch_size'] = 32

        # Check if there is already an entry for this combiner
        try:
            self.data = self.coll.find_one({'id': self.id})
            if not self.data:
                self.data = {
                    'id': self.id,
                    'model_id': config['seedmodel']
                }
                result = self.coll.insert_one(self.data)
        except:
            # TODO: Look up in the hierarchy for a global model
            self.data = {
                'model_id': config['seedmodel']
            }

        print("ORCHESTRATOR starting from model {}".format(self.data['model_id']))
        self.__set_model(self.data['model_id'])

        import time

        ready = False
        while not ready:
            active = self.nr_active_trainers()
            if active >= config['active_clients']:
                ready = True
            else:
                print("waiting for {} clients to get started, currently: {}".format(config['active_clients'] - active,
                                                                                    active), flush=True)
            time.sleep(1)

        for r in range(1, config['rounds'] + 1):
            print("STARTING ROUND {}".format(r), flush=True)
            print("\t Starting training round {}".format(r), flush=True)

            self.__assign_clients(config['active_clients'])

            model = self.__training_round()
            if model:
                model_name = "/model/"+"global_model_"+str(r)+".h5"
                print("\t Training round completed.", flush=True)
                fod, outfile_name = tempfile.mkstemp(suffix='.h5')
                model.save(model_name)
                model.save(outfile_name)
                # Upload new model to storage repository
                model_id = self.storage.set_model(outfile_name, is_file=True)
                os.unlink(outfile_name)

                # And update the db record
                self.__set_model(model_id)
                print("...done. New global model: {}".format(self.model_id))

                print("\t Starting validation round {}".format(r))
                self.__validation_round()

                print("------------------------------------------")
                print("ROUND COMPLETED.", flush=True)
                print("\n")
            else:
                print("\t Failed to update global model in round {0}!".format(r))
