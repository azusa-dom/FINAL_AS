--- a/src/train.py
+++ b/src/train.py
@@
-from .models import get_model
-from .models import get_model, ClinicalMLP
+from .models import get_model, ClinicalMLP, ResNet18Encoder

 def train_one_epoch(model, loader, optimizer, device):
@@
-    for img, clin, label in loader:
+    for img, clin, label in loader:
         img, clin, label = img.to(device), clin.to(device), label.to(device)
         optimizer.zero_grad()
-        if (
-        if isinstance(model, ClinicalMLP):
-            output = model(clin)
-        elif (
-            isinstance(model, nn.Module)
-            and len(model.forward.__code__.co_varnames) == 3
-        ):
-            output = model(img, clin)
-        else:
-            output = model(img)
+        # —— 根据模型类型决定要喂哪个输入 —— #
+        if isinstance(model, ClinicalMLP):
+            # 纯临床
+            output = model(clin)
+        elif isinstance(model, ResNet18Encoder):
+            # 纯图像
+            output = model(img)
+        else:
+            # 融合模型
+            output = model(img, clin)
