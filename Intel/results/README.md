**The following entries are copied from Intel v0.7 submissions:**

 - 1-node-4s-8380H-pytorch 
 - 1-node-8s-8380H-mxnet
 - 1-node-8s-8380H-tensorflow 
 - 1-node-8s-8380H-pytorch
 - 2-nodes-8s-8380H-pytorch 
 - 4-nodes-16s-8380H-pytorch
 - 8-nodes-32s-8380H-tensorflow

Their logs are appended with the one line below to pass the v1.0 submission checker.

    :::MLLOG {"namespace": "worker0", "time_ms": 0, "event_type": "POINT_IN_TIME", "key": "gradient_accumulation_steps", "value": 1, "metadata": {}}


**The following entries are new v1.0 submissions:**
 
 - 4-nodes-32s-8376H-pytorch	   
 - 8-nodes-64s-8376H-pytorch
 - 4-nodes-32s-8376H-tensorflow  
 - 8-nodes-64s-8376H-tensorflow
