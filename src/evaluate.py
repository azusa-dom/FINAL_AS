def evaluate_checkpoint(
    model_name: str,
    ckpt_path: Path,
    csv: Path,
    img_dir: Path,
    mode: str,
    out_dir: Path,
):
    loader = create_dataloader(
        csv,
        img_dir,
        batch_size=16,
        train=False,
        augment=False,
        mode=mode,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, prob = [], []
    with torch.no_grad():
        for img, clin, label in loader:
            img, clin = img.to(device), clin.to(device)

            # —— 根据模型名称决定输入 —— #
            if model_name == "clin_only":
                # 纯临床模型，只用临床特征
                output = model(clin)
            elif model_name == "mri_only":
                # 纯图像模型，只用图像
                output = model(img)
            else:
                # 其余融合模型（early_fusion／late_fusion_features）
                output = model(img, clin)

            prob.append(torch.softmax(output, 1)[:, 1].cpu())
            y_true.append(label)

    prob = torch.cat(prob).numpy()
    y_true = torch.cat(y_true).numpy()

    # 后面不变，计算指标并保存
    roc = roc_auc_score(y_true, prob)
    pr = average_precision_score(y_true, prob)
    brier = brier_score_loss(y_true, prob)
    acc = accuracy_score(y_true, prob > 0.5)
    sens = recall_score(y_true, prob > 0.5)
    cm = confusion_matrix(y_true, prob > 0.5)
    if cm.size == 4:
        tn, fp, _, _ = cm.ravel()
        spec = tn / (tn + fp)
    else:
        spec = 0.0
    roc_ci = bootstrap_ci(roc_auc_score, y_true, prob)
    pr_ci = bootstrap_ci(average_precision_score, y_true, prob)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_pr(y_true, prob, str(out_dir / "curve"))
    plot_calibration_curve(y_true, prob, str(out_dir / "calibration.png"))
    decision_curve_analysis(y_true, prob, str(out_dir / "dca.png"))

    metrics = {
        'roc_auc': roc,
        'pr_auc': pr,
        'brier': brier,
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'roc_auc_ci': roc_ci.tolist(),
        'pr_auc_ci': pr_ci.tolist(),
    }
    save_json(metrics, out_dir / 'test_metrics.json')
    print(json.dumps(metrics, indent=2))
