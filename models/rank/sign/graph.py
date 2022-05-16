# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    This package implement Graph structure for handling graph data.
"""

import os
import json
import copy
import warnings
from collections import defaultdict

import numpy as np
import paddle

from pgl.utils import op
import pgl.graph_kernel as graph_kernel
from pgl.message import Message
from pgl.utils.edge_index import EdgeIndex
from pgl.utils.helper import check_is_tensor, scatter, maybe_num_nodes
from pgl.utils.helper import generate_segment_id_from_index, unique_segment

try:
    from paddle.incubate import graph_send_recv
except:
    from pgl.utils.helper import graph_send_recv


class Graph(object):
    """Implementation of graph interface in pgl.

    This is a simple implementation of graph structure in pgl. 

    `pgl.Graph` is an alias for `pgl.graph.Graph` 

    Args:

        edges: list of (u, v) tuples, 2D numpy.ndarray or 2D paddle.Tensor. 

        num_nodes (optional: int, numpy or paddle.Tensor): Number of nodes in a graph. 
                           If not provided, the number of nodes will be infered from edges. 

        node_feat (optional): a dict of numpy array as node features.

        edge_feat (optional): a dict of numpy array as edge features (should
                                have consistent order with edges).

    Examples 1:

        - Create a graph with numpy.
        - Convert it into paddle.Tensor.
        - Do send recv for graph neural network.

        .. code-block:: python

            import numpy as np
            import pgl

            num_nodes = 5
            edges = [ (0, 1), (1, 2), (3, 4)]
            feature = np.random.randn(5, 100).astype(np.float32)
            edge_feature = np.random.randn(3, 100).astype(np.float32)
            graph = pgl.Graph(num_nodes=num_nodes,
                        edges=edges,
                        node_feat={
                            "feature": feature
                        },
                        edge_feat={
                            "edge_feature": edge_feature
                        })
            graph.tensor() 

            model = pgl.nn.GCNConv(100, 100)
            out = model(graph, graph.node_feat["feature"])
          

    Examples 2: 

        - Create a graph with paddle.Tensor.
        - Do send recv for graph neural network.

        .. code-block:: python

            import paddle 
            import pgl

            num_nodes = 5
            edges = paddle.to_tensor([ (0, 1), (1, 2), (3, 4)])
            feature = paddle.randn(shape=[5, 100])
            edge_feature = paddle.randn(shape=[3, 100])
            graph = pgl.Graph(num_nodes=num_nodes,
                        edges=edges,
                        node_feat={
                            "feature": feature
                        },
                        edge_feat={
                            "edge_feature": edge_feature
                        })

            model = pgl.nn.GCNConv(100, 100)
            out = model(graph, graph.node_feat["feature"])

    """

    def __init__(self,
                 edges,
                 num_nodes=None,
                 node_feat=None,
                 edge_feat=None,
                 **kwargs):
        if node_feat is not None:
            self._node_feat = node_feat
        else:
            self._node_feat = {}

        if edge_feat is not None:
            self._edge_feat = edge_feat
        else:
            self._edge_feat = {}

        if not check_is_tensor(edges):
            if isinstance(edges, np.ndarray):
                if edges.dtype != "int64":
                    edges = edges.astype("int64")
            else:
                edges = np.array(edges, dtype="int64")

        self._edges = edges

        if num_nodes is None:
            self._num_nodes = maybe_num_nodes(self._edges)
        else:
            self._num_nodes = num_nodes

        self._adj_src_index = kwargs.get("adj_src_index", None)
        self._adj_dst_index = kwargs.get("adj_dst_index", None)

        if check_is_tensor(self._num_nodes, self._edges,
                           *list(self._node_feat.values()),
                           *list(self._edge_feat.values())):
            self._is_tensor = True
        elif self._adj_src_index is not None and self._adj_src_index.is_tensor(
        ):
            self._is_tensor = True
        elif self._adj_dst_index is not None and self._adj_dst_index.is_tensor(
        ):
            self._is_tensor = True
        else:
            self._is_tensor = False

        if self._is_tensor:
            # ensure all variable is tenosr
            if not check_is_tensor(self._num_nodes):
                self._num_nodes = paddle.to_tensor(self._num_nodes)

            if not check_is_tensor(self._edges):
                self._edges = paddle.to_tensor(self._edges)

            for key in self._node_feat:
                if not check_is_tensor(self._node_feat[key]):
                    self._node_feat[key] = paddle.to_tensor(self._node_feat[
                        key])

            for key in self._edge_feat:
                if not check_is_tensor(self._edge_feat[key]):
                    self._edge_feat[key] = paddle.to_tensor(self._edge_feat[
                        key])

            if self._adj_src_index is not None:
                if not self._adj_src_index.is_tensor():
                    self._adj_src_index.tensor(inplace=True)

            if self._adj_dst_index is not None:
                if not self._adj_dst_index.is_tensor():
                    self._adj_dst_index.tensor(inplace=True)

        # preprocess graph level informations
        self._process_graph_info(**kwargs)
        self._nodes = None

    def recv(self, reduce_func, msg, recv_mode="dst"):
        """Recv message and aggregate the message by reduce_func.

        The UDF reduce_func function should has the following format.

        .. code-block:: python

            def reduce_func(msg):
                '''
                    Args:

                        msg: An instance of Message class.

                    Return:

                        It should return a tensor with shape (batch_size, out_dims).
                '''
                pass

        Args:

            msg: A dictionary of tensor created by send function..

            reduce_func: A callable UDF reduce function.

        Return:

            A tensor with shape (num_nodes, out_dims). The output for nodes with 
            no message will be zeros.

        """

        if not self._is_tensor:
            raise ValueError("You must call Graph.tensor()")

        if not isinstance(msg, dict):
            raise TypeError(
                "The input of msg should be a dict, but receives a %s" %
                (type(msg)))

        if not callable(reduce_func):
            raise TypeError("reduce_func should be callable")

        src, dst, eid = self.sorted_edges(sort_by=recv_mode)

        msg = op.RowReader(msg, eid)

        if (recv_mode == "dst") and (not hasattr(self, "_dst_uniq_ind")):
            self._dst_uniq_ind, self._dst_segment_ids = unique_segment(dst)
        if (recv_mode == "src") and (not hasattr(self, "_src_uniq_ind")):
            self._src_uniq_ind, self._src_segment_ids = unique_segment(src)

        if recv_mode == "dst":
            uniq_ind, segment_ids = self._dst_uniq_ind, self._dst_segment_ids
        elif recv_mode == "src":
            uniq_ind, segment_ids = self._src_uniq_ind, self._src_segment_ids

        bucketed_msg = Message(msg, segment_ids)
        output = reduce_func(bucketed_msg)
        output_dim = output.shape[-1]
        init_output = paddle.zeros(
            shape=[self._num_nodes, output_dim], dtype=output.dtype)
        final_output = scatter(init_output, uniq_ind, output)

        return final_output

    @classmethod
    def load(cls, path, mmap_mode="r"):
        """Load Graph from path and return a Graph in numpy. 

        Args:

            path: The directory path of the stored Graph.

            mmap_mode: Default :code:`mmap_mode="r"`. If not None, memory-map the graph.  

        """

        num_nodes = np.load(
            os.path.join(path, 'num_nodes.npy'), mmap_mode=mmap_mode)
        edges = np.load(os.path.join(path, 'edges.npy'), mmap_mode=mmap_mode)
        num_graph = np.load(
            os.path.join(path, 'num_graph.npy'), mmap_mode=mmap_mode)
        if os.path.exists(os.path.join(path, 'graph_node_index.npy')):
            graph_node_index = np.load(
                os.path.join(path, 'graph_node_index.npy'),
                mmap_mode=mmap_mode)
        else:
            graph_node_index = None

        if os.path.exists(os.path.join(path, 'graph_edge_index.npy')):
            graph_edge_index = np.load(
                os.path.join(path, 'graph_edge_index.npy'),
                mmap_mode=mmap_mode)
        else:
            graph_edge_index = None

        if os.path.isdir(os.path.join(path, 'adj_src')):
            adj_src_index = EdgeIndex.load(
                os.path.join(path, 'adj_src'), mmap_mode=mmap_mode)
        else:
            adj_src_index = None

        if os.path.isdir(os.path.join(path, 'adj_dst')):
            adj_dst_index = EdgeIndex.load(
                os.path.join(path, 'adj_dst'), mmap_mode=mmap_mode)
        else:
            adj_dst_index = None

        def _load_feat(feat_path):
            """Load features from .npy file.
            """
            feat = {}
            if os.path.isdir(feat_path):
                for feat_name in os.listdir(feat_path):
                    feat[os.path.splitext(feat_name)[0]] = np.load(
                        os.path.join(feat_path, feat_name),
                        mmap_mode=mmap_mode)
            return feat

        node_feat = _load_feat(os.path.join(path, 'node_feat'))
        edge_feat = _load_feat(os.path.join(path, 'edge_feat'))
        return cls(edges=edges,
                   num_nodes=num_nodes,
                   node_feat=node_feat,
                   edge_feat=edge_feat,
                   adj_src_index=adj_src_index,
                   adj_dst_index=adj_dst_index,
                   _num_graph=num_graph,
                   _graph_node_index=graph_node_index,
                   _graph_edge_index=graph_edge_index)

    def is_tensor(self):
        """Return whether the Graph is in paddle.Tensor or numpy format.
        """
        return self._is_tensor

    def _apply_to_tensor(self, key, value, inplace=True):
        if value is None:
            return value

        if key == '_is_tensor':
            # set is_tensor to True
            return True

        if isinstance(value, EdgeIndex):
            value = value.tensor(inplace=inplace)

        elif isinstance(value, dict):
            if inplace:
                for k, v in value.items():
                    value[k] = paddle.to_tensor(v)
            else:
                new_value = {}
                for k, v in value.items():
                    new_value[k] = paddle.to_tensor(v)
                value = new_value
        else:
            value = paddle.to_tensor(value)
        return value

    def tensor(self, inplace=True):
        """Convert the Graph into paddle.Tensor format.

        In paddle.Tensor format, the graph edges and node features are in paddle.Tensor format.
        You can use send and recv in paddle.Tensor graph.
        
        Args:

            inplace: (Default True) Whether to convert the graph into tensor inplace. 
        
        """

        if self._is_tensor:
            return self

        if inplace:
            for key in self.__dict__:
                self.__dict__[key] = self._apply_to_tensor(
                    key, self.__dict__[key], inplace)
            return self
        else:
            new_dict = {}
            for key in self.__dict__:
                new_dict[key] = self._apply_to_tensor(key, self.__dict__[key],
                                                      inplace)

            graph = self.__class__(
                num_nodes=new_dict["_num_nodes"],
                edges=new_dict["_edges"],
                node_feat=new_dict["_node_feat"],
                edge_feat=new_dict["_edge_feat"],
                adj_src_index=new_dict["_adj_src_index"],
                adj_dst_index=new_dict["_adj_dst_index"],
                **new_dict)
            return graph

    def _apply_to_numpy(self, key, value, inplace=True):
        if value is None:
            return value

        if key == '_is_tensor':
            # set is_tensor to True
            return False

        if isinstance(value, EdgeIndex):
            value = value.numpy(inplace=inplace)
        elif isinstance(value, dict):
            if inplace:
                for k, v in value.items():
                    value[k] = v.numpy()
            else:
                new_value = {}
                for k, v in value.items():
                    new_value[k] = v.numpy()
                value = new_value
        else:
            value = value.numpy()
        return value

    def numpy(self, inplace=True):
        """Convert the Graph into numpy format.

        In numpy format, the graph edges and node features are in numpy.ndarray format.
        But you can't use send and recv in numpy graph.
        
        Args:

            inplace: (Default True) Whether to convert the graph into numpy inplace. 
        
        """
        if not self._is_tensor:
            return self

        if inplace:
            for key in self.__dict__:
                self.__dict__[key] = self._apply_to_numpy(
                    key, self.__dict__[key], inplace)
            return self
        else:
            new_dict = {}
            for key in self.__dict__:
                new_dict[key] = self._apply_to_numpy(key, self.__dict__[key],
                                                     inplace)

            graph = self.__class__(
                num_nodes=new_dict["_num_nodes"],
                edges=new_dict["_edges"],
                node_feat=new_dict["_node_feat"],
                edge_feat=new_dict["_edge_feat"],
                adj_src_index=new_dict["_adj_src_index"],
                adj_dst_index=new_dict["_adj_dst_index"],
                **new_dict)
            return graph

    def dump(self, path):
        """Dump the graph into a directory.

        This function will dump the graph information into the given directory path. 
        The graph can be read back with :code:`pgl.Graph.load`

        Args:
            path: The directory for the storage of the graph.

        """
        if self._is_tensor:
            # Convert back into numpy and dump.
            graph = self.numpy(inplace=False)
            graph.dump(path)
        else:
            if not os.path.exists(path):
                os.makedirs(path)

            np.save(os.path.join(path, 'num_nodes.npy'), self._num_nodes)
            np.save(os.path.join(path, 'edges.npy'), self._edges)
            np.save(os.path.join(path, 'num_graph.npy'), self._num_graph)

            if self._adj_src_index is not None:
                self._adj_src_index.dump(os.path.join(path, 'adj_src'))

            if self._adj_dst_index is not None:
                self._adj_dst_index.dump(os.path.join(path, 'adj_dst'))

            if self._graph_node_index is not None:
                np.save(
                    os.path.join(path, 'graph_node_index.npy'),
                    self._graph_node_index)

            if self._graph_edge_index is not None:
                np.save(
                    os.path.join(path, 'graph_edge_index.npy'),
                    self._graph_edge_index)

            def _dump_feat(feat_path, feat):
                """Dump all features to .npy file.
                """
                if len(feat) == 0:
                    return

                if not os.path.exists(feat_path):
                    os.makedirs(feat_path)

                for key in feat:
                    value = feat[key]
                    np.save(os.path.join(feat_path, key + ".npy"), value)

            _dump_feat(os.path.join(path, "node_feat"), self.node_feat)
            _dump_feat(os.path.join(path, "edge_feat"), self.edge_feat)

    @property
    def adj_src_index(self):
        """Return an EdgeIndex object for src.
        """
        if self._adj_src_index is None:
            u = self._edges[:, 0]
            v = self._edges[:, 1]
            self._adj_src_index = EdgeIndex.from_edges(
                u=u, v=v, num_nodes=self._num_nodes)
        return self._adj_src_index

    @property
    def adj_dst_index(self):
        """Return an EdgeIndex object for dst.
        """
        if self._adj_dst_index is None:
            v = self._edges[:, 0]
            u = self._edges[:, 1]
            self._adj_dst_index = EdgeIndex.from_edges(
                u=u, v=v, num_nodes=self._num_nodes)
        return self._adj_dst_index

    @property
    def edge_feat(self):
        """Return a dictionary of edge features.
        """
        return self._edge_feat

    @property
    def node_feat(self):
        """Return a dictionary of node features.
        """
        return self._node_feat

    @property
    def num_edges(self):
        """Return the number of edges.
        """
        if self._is_tensor:
            return paddle.shape(self._edges)[0]
        else:
            return self._edges.shape[0]

    @property
    def num_nodes(self):
        """Return the number of nodes.
        """
        return self._num_nodes

    @property
    def edges(self):
        """Return all edges in numpy.ndarray or paddle.Tensor with shape (num_edges, 2).
        """
        return self._edges

    def sorted_edges(self, sort_by="src"):
        """Return sorted edges with different strategies.

        This function will return sorted edges with different strategy.
        If :code:`sort_by="src"`, then edges will be sorted by :code:`src`
        nodes and otherwise :code:`dst`.

        Args:

            sort_by: The type for sorted edges. ("src" or "dst")

        Return:

            A tuple of (sorted_src, sorted_dst, sorted_eid).
        """
        if sort_by not in ["src", "dst"]:
            raise ValueError("sort_by should be in 'src' or 'dst'.")
        if sort_by == 'src':
            src, dst, eid = self.adj_src_index.triples()
        else:
            dst, src, eid = self.adj_dst_index.triples()
        return src, dst, eid

    @property
    def nodes(self):
        """Return all nodes id from 0 to :code:`num_nodes - 1`
        """
        if self._nodes is None:
            if self.is_tensor():
                self._nodes = paddle.arange(self.num_nodes)
            else:
                self._nodes = np.arange(self.num_nodes)
        return self._nodes

    def indegree(self, nodes=None):
        """Return the indegree of the given nodes

        This function will return indegree of given nodes.

        Args:

            nodes: Return the indegree of given nodes,
                   if nodes is None, return indegree for all nodes

        Return:

            A numpy.ndarray or paddle.Tensor as the given nodes' indegree.
        """
        if nodes is None:
            return self.adj_dst_index.degree
        else:
            if self._is_tensor:
                return paddle.gather(self.adj_dst_index.degree, nodes)
            else:
                return self.adj_dst_index.degree[nodes]

    def outdegree(self, nodes=None):
        """Return the outdegree of the given nodes.

        This function will return outdegree of given nodes.

        Args:

            nodes: Return the outdegree of given nodes,
                   if nodes is None, return outdegree for all nodes

        Return:

            A numpy.array or paddle.Tensor as the given nodes' outdegree.
        """
        if nodes is None:
            return self.adj_src_index.degree
        else:
            if self._is_tensor:
                return paddle.gather(self.adj_src_index.degree, nodes)
            else:
                return self.adj_src_index.degree[nodes]

    def successor(self, nodes=None, return_eids=False):
        """Find successor of given nodes.

        This function will return the successor of given nodes.

        Args:

            nodes: Return the successor of given nodes,
                   if nodes is None, return successor for all nodes.

            return_eids: If True return nodes together with corresponding eid

        Return:

            Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of successor ids for given nodes. If :code:`return_eids=True`, there will
            be an additional list of numpy.ndarray and each numpy.ndarray represent
            a list of eids that connected nodes to their successors.

        Example:
            .. code-block:: python

                import numpy as np
                import pgl

                num_nodes = 5
                edges = [ (0, 1), (1, 2), (3, 4)]
                graph = pgl.Graph(num_nodes=num_nodes,
                        edges=edges)
                succ, succ_eid = graph.successor(return_eids=True)

            This will give output.

            .. code-block:: python

                succ:
                      [[1],
                       [2],
                       [],
                       [4],
                       []]

                succ_eid:
                      [[0],
                       [1],
                       [],
                       [2],
                       []]

        """
        if self.is_tensor():
            raise ValueError(
                "You must call Graph.numpy() first. Tensor object don't supprt successor now."
            )
        else:
            if return_eids:
                return self.adj_src_index.view_v(
                    nodes), self.adj_src_index.view_eid(nodes)
            else:
                return self.adj_src_index.view_v(nodes)

    def sample_successor(self,
                         nodes,
                         max_degree,
                         return_eids=False,
                         shuffle=False):
        """Sample successors of given nodes.

        Args:

            nodes: Given nodes whose successors will be sampled.

            max_degree: The max sampled successors for each nodes.

            return_eids: Whether to return the corresponding eids.

        Return:

            Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of sampled successor ids for given nodes. If :code:`return_eids=True`, there will
            be an additional list of numpy.ndarray and each numpy.ndarray represent
            a list of eids that connected nodes to their successors.
        """
        if self.is_tensor():
            raise ValueError(
                "You must call Graph.numpy() first. Tensor object don't supprt sample_successor now."
            )
        else:
            node_succ = self.successor(nodes, return_eids=return_eids)
            if return_eids:
                node_succ, node_succ_eid = node_succ

            if nodes is None:
                nodes = self.nodes

            node_succ = node_succ.tolist()

            if return_eids:
                node_succ_eid = node_succ_eid.tolist()

            if return_eids:
                return graph_kernel.sample_subset_with_eid(
                    node_succ, node_succ_eid, max_degree, shuffle)
            else:
                return graph_kernel.sample_subset(node_succ, max_degree,
                                                  shuffle)

    def predecessor(self, nodes=None, return_eids=False):
        """Find predecessor of given nodes.

        This function will return the predecessor of given nodes.

        Args:

            nodes: Return the predecessor of given nodes,
                   if nodes is None, return predecessor for all nodes.

            return_eids: If True return nodes together with corresponding eid

        Return:

            Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of predecessor ids for given nodes. If :code:`return_eids=True`, there will
            be an additional list of numpy.ndarray and each numpy.ndarray represent
            a list of eids that connected nodes to their predecessors.

        Example:

            .. code-block:: python

                import numpy as np
                import pgl
 
                num_nodes = 5
                edges = [ (0, 1), (1, 2), (3, 4)]
                graph = pgl.Graph(num_nodes=num_nodes,
                        edges=edges)
                pred, pred_eid = graph.predecessor(return_eids=True)

            This will give output.

            .. code-block:: python

                pred:
                      [[],
                       [0],
                       [1],
                       [],
                       [3]]

                pred_eid:
                      [[],
                       [0],
                       [1],
                       [],
                       [2]]

        """
        if self.is_tensor():
            raise ValueError(
                "You must call Graph.numpy() first. Tensor object don't supprt predecessor now."
            )
        else:
            if return_eids:
                return self.adj_dst_index.view_v(
                    nodes), self.adj_dst_index.view_eid(nodes)
            else:
                return self.adj_dst_index.view_v(nodes)

    def sample_predecessor(self,
                           nodes,
                           max_degree,
                           return_eids=False,
                           shuffle=False):
        """Sample predecessor of given nodes.

        Args:

            nodes: Given nodes whose predecessor will be sampled.

            max_degree: The max sampled predecessor for each nodes.

            return_eids: Whether to return the corresponding eids.

        Return:

            Return a list of numpy.ndarray and each numpy.ndarray represent a list
            of sampled predecessor ids for given nodes. If :code:`return_eids=True`, there will
            be an additional list of numpy.ndarray and each numpy.ndarray represent
            a list of eids that connected nodes to their predecessors.
        """
        if self.is_tensor():
            raise ValueError(
                "You must call Graph.numpy() first. Tensor object don't supprt sample_predecessor now."
            )
        else:
            node_pred = self.predecessor(nodes, return_eids=return_eids)
            if return_eids:
                node_pred, node_pred_eid = node_pred

            if nodes is None:
                nodes = self.nodes

            node_pred = node_pred.tolist()

            if return_eids:
                node_pred_eid = node_pred_eid.tolist()

            if return_eids:
                return graph_kernel.sample_subset_with_eid(
                    node_pred, node_pred_eid, max_degree, shuffle)
            else:
                return graph_kernel.sample_subset(node_pred, max_degree,
                                                  shuffle)

    @property
    def num_graph(self):
        """ Return Number of Graphs"""
        return self._num_graph

    @property
    def graph_node_id(self):
        """ Return a numpy.ndarray or paddle.Tensor with shape [num_nodes] 
        that indicates which graph the nodes belongs to.

        Examples:

        .. code-block:: python
       
            import numpy as np
            import pgl

            num_nodes = 5
            edges = [ (0, 1), (1, 2), (3, 4)]
            graph = pgl.Graph(num_nodes=num_nodes,
                        edges=edges)
            joint_graph = pgl.Graph.batch([graph, graph])
            print(joint_graph.graph_node_id)

            >>> [0, 0, 0, 0, 0, 1, 1, 1, 1 ,1]
 
        """

        return generate_segment_id_from_index(self._graph_node_index)

    @property
    def graph_edge_id(self):
        """ Return a numpy.ndarray or paddle.Tensor with shape [num_edges] 
        that indicates which graph the edges belongs to.

        Examples:

        .. code-block:: python
       
            import numpy as np
            import pgl

            num_nodes = 5
            edges = [ (0, 1), (1, 2), (3, 4)]
            graph = pgl.Graph(num_nodes=num_nodes,
                        edges=edges)
            joint_graph = pgl.Graph.batch([graph, graph])
            print(joint_graph.graph_edge_id)

            >>> [0, 0, 0, 1, 1, 1]
 
        """

        return generate_segment_id_from_index(self._graph_edge_index)

    def send_recv(self, feature, reduce_func="sum"):
        """This method combines the send and recv function using graph_send_recv API.

        Now, this method only supports default copy send function, and built-in receive 
        function ('sum', 'mean', 'max', 'min').

        Args:

           feature (Tensor): The node feature of a graph.

           reduce_func (str): Difference reduce function, including 'sum', 'mean', 'max', 'min'.

        """

        assert isinstance(feature, paddle.Tensor) or isinstance(feature, paddle.fluid.framework.Variable), \
            "The input of send_recv method should be Tensor."

        assert reduce_func in ['sum', 'mean', 'max', 'min'], \
            "Only support 'sum', 'mean', 'max', 'min' built-in reduce functions."

        src, dst = self.edges[:, 0], self.edges[:, 1]
        return graph_send_recv(feature, src, dst, pool_type=reduce_func)

    def send(
            self,
            message_func,
            src_feat=None,
            dst_feat=None,
            edge_feat=None, ):
        """Send message from all src nodes to dst nodes.

        The UDF message function should has the following format.

        .. code-block:: python

            def message_func(src_feat, dst_feat, edge_feat):
                '''
                    Args:
                        src_feat: the node feat dict attached to the src nodes.
                        dst_feat: the node feat dict attached to the dst nodes.
                        edge_feat: the edge feat dict attached to the
                                   corresponding (src, dst) edges.

                    Return:
                        It should return a tensor or a dictionary of tensor. And each tensor
                        should have a shape of (num_edges, dims).
                '''
                return {'msg': src_feat['h']}

        Args:
            message_func: UDF function.
            src_feat: a dict {name: tensor,} to build src node feat
            dst_feat: a dict {name: tensor,} to build dst node feat
            node_feat: a dict {name: tensor,} to build both src and dst node feat
            edge_feat: a dict {name: tensor,} to build edge feat

        Return:
            A dictionary of tensor representing the message. Each of the values
            in the dictionary has a shape (num_edges, dim) which should be collected
            by :code:`recv` function.
        """
        msg = {}
        if self._is_tensor:

            src_feat_temp = {}
            dst_feat_temp = {}
            if src_feat is not None:
                assert isinstance(src_feat,
                                  dict), "The input src_feat must be a dict"
                src_feat_temp.update(src_feat)

            if dst_feat is not None:
                assert isinstance(dst_feat,
                                  dict), "The input dst_feat must be a dict"
                dst_feat_temp.update(dst_feat)

            edge_feat_temp = {}
            if edge_feat is not None:
                assert isinstance(edge_feat,
                                  dict), "The input edge_feat must be a dict"
                edge_feat_temp.update(edge_feat)

            src = self.edges[:, 0]
            dst = self.edges[:, 1]

            src_feat = op.RowReader(src_feat_temp, src)
            dst_feat = op.RowReader(dst_feat_temp, dst)
            msg = message_func(src_feat, dst_feat, edge_feat_temp)

            if not isinstance(msg, dict):
                raise TypeError(
                    "The outputs of the %s function is expected to be a dict, but got %s" \
                            % (message_func.__name__, type(msg)))
            return msg
        else:
            raise ValueError("You must call Graph.tensor() first")
        return msg

    def _process_graph_info(self, **kwargs):
        if ("_graph_node_index" in kwargs) and (
                kwargs["_graph_node_index"] is not None):
            self._graph_node_index = kwargs["_graph_node_index"]
        else:
            self._graph_node_index = None

        if ("_graph_edge_index" in kwargs) and (
                kwargs["_graph_edge_index"] is not None):
            self._graph_edge_index = kwargs["_graph_edge_index"]
        else:
            self._graph_edge_index = None

        if ("_num_graph" in kwargs) and (kwargs["_num_graph"] is not None):
            self._num_graph = kwargs["_num_graph"]
        else:
            if self._is_tensor:
                self._num_graph = paddle.ones(shape=[1], dtype="int32")
                self._graph_node_index = paddle.concat([
                    paddle.zeros(
                        shape=[1], dtype="int32"), paddle.full(
                            shape=[1],
                            fill_value=self.num_nodes,
                            dtype="int32")
                ])
                self._graph_edge_index = paddle.concat([
                    paddle.zeros(
                        shape=[1], dtype="int32"), paddle.full(
                            shape=[1],
                            fill_value=self.num_edges,
                            dtype="int32")
                ])
            else:
                self._num_graph = 1
                self._graph_node_index = np.array(
                    [0, self._num_nodes], dtype="int64")
                self._graph_edge_index = np.array(
                    [0, self.num_edges], dtype="int64")

    @classmethod
    def disjoint(cls, graph_list, merged_graph_index=False):
        """This method disjoint list of graph into a big graph.

        Args:

            graph_list (Graph List): A list of Graphs.

            merged_graph_index: whether to keeped the graph_id that the nodes belongs to.

       
        .. code-block:: python
       
            import numpy as np
            import pgl

            num_nodes = 5
            edges = [ (0, 1), (1, 2), (3, 4)]
            graph = pgl.Graph(num_nodes=num_nodes,
                        edges=edges)
            joint_graph = pgl.Graph.disjoint([graph, graph], merged_graph_index=False)
            print(joint_graph.graph_node_id)
            >>> [0, 0, 0, 0, 0, 1, 1, 1, 1 ,1]
            print(joint_graph.num_graph)
            >>> 2

            joint_graph = pgl.Graph.disjoint([graph, graph], merged_graph_index=True)
            print(joint_graph.graph_node_id)
            >>> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            print(joint_graph.num_graph)
            >>> 1 
        """
        # TODO:@Yelrose supporting disjoint a disjointed graph_list.
        assert len(
            graph_list
        ) > 0, "The input graph_list of Graph.disjoint has length %s. It should be greater than 0. " % len(
            graph_list)

        is_tensor = graph_list[0].is_tensor()

        edges = cls._join_edges(graph_list)
        num_nodes = cls._join_nodes(graph_list)
        node_feat = cls._join_feature(graph_list, mode="node")
        edge_feat = cls._join_feature(graph_list, mode="edge")

        if merged_graph_index is True:
            num_graph = None
            graph_node_index = None
            graph_edge_index = None
        else:
            num_graph = paddle.to_tensor([len(graph_list)], "int64") \
                    if is_tensor else len(graph_list)
            graph_node_index = cls._join_graph_index(graph_list, mode="node")
            graph_edge_index = cls._join_graph_index(graph_list, mode="edge")

        graph = cls(num_nodes=num_nodes,
                    edges=edges,
                    node_feat=node_feat,
                    edge_feat=edge_feat,
                    _num_graph=num_graph,
                    _graph_node_index=graph_node_index,
                    _graph_edge_index=graph_edge_index)
        return graph

    @staticmethod
    def batch(graph_list):
        """This is alias on `pgl.Graph.disjoint` with merged_graph_index=False"""
        return Graph.disjoint(graph_list, merged_graph_index=False)

    @staticmethod
    def _join_graph_index(graph_list, mode="node"):
        is_tensor = graph_list[0].is_tensor()
        if mode == "node":
            counts = [g.num_nodes for g in graph_list]
        elif mode == "edge":
            counts = [g.num_edges for g in graph_list]
        else:
            raise ValueError(
                "mode must be in ['node', 'edge']. But received model=%s" %
                mode)

        if is_tensor:
            counts = paddle.concat(counts)
        return op.get_index_from_counts(counts)

    @staticmethod
    def _join_nodes(graph_list):
        num_nodes = 0
        for g in graph_list:
            num_nodes = g.num_nodes + num_nodes
        return num_nodes

    @staticmethod
    def _join_feature(graph_list, mode="node"):
        """join node features for multiple graph"""
        is_tensor = graph_list[0].is_tensor()
        feat = defaultdict(lambda: [])
        if mode == "node":
            for graph in graph_list:
                for key in graph.node_feat:
                    feat[key].append(graph.node_feat[key])
        elif mode == "edge":
            for graph in graph_list:
                for key in graph.edge_feat:
                    feat[key].append(graph.edge_feat[key])
        else:
            raise ValueError(
                "mode must be in ['node', 'edge']. But received model=%s" %
                mode)

        ret_feat = {}
        for key in feat:
            if len(feat[key]) == 1:
                ret_feat[key] = feat[key][0]
            else:
                if is_tensor:
                    ret_feat[key] = paddle.concat(feat[key], 0)
                else:
                    ret_feat[key] = np.concatenate(feat[key], axis=0)
        return ret_feat

    @staticmethod
    def _join_edges(graph_list):
        """join edges for multiple graph"""
        is_tensor = graph_list[0].is_tensor()
        list_edges = []
        start_offset = 0
        for graph in graph_list:
            edges = graph.edges
            if len(edges) > 0:
                edges = edges + start_offset
                list_edges.append(edges)
            start_offset += graph.num_nodes
        if len(list_edges) == 1:
            return list_edges[0]

        if is_tensor:
            edges = paddle.concat(list_edges, 0)
        else:
            edges = np.concatenate(list_edges, axis=0)
        return edges

    def node_batch_iter(self, batch_size, shuffle=True):
        """Node batch iterator

        Iterate all node by batch.

        Args:
            batch_size: The batch size of each batch of nodes.

            shuffle: Whether shuffle the nodes.

        Return:
            Batch iterator
        """
        if self.is_tensor():
            if shuffle:
                perm = paddle.randperm(self.num_nodes)
            else:
                perm = paddle.arange(self.num_nodes)
        else:
            perm = np.arange(self.num_nodes)
            if shuffle:
                np.random.shuffle(perm)

        start = 0
        while start < self.num_nodes:
            yield perm[start:start + batch_size]
            start += batch_size

    def to_mmap(self, path="./tmp"):
        """Turn the Graph into Memmap mode which can share memory between processes.
        """
        self.dump(path)
        graph = Graph.load(path, mmap_mode="r")
        return graph
