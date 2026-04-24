"""
backend/app/api/reports.py
===========================
Endpoints for drift report, model comparison, and PDF clinical report.
"""
import json
import logging
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
router = APIRouter()

BASE = Path("/app")
PROC_DIR = BASE / "data" / "processed"


# ── 1. Drift Report ───────────────────────────────────────────
@router.get("/drift/summary")
async def get_drift_summary():
    """Returns drift detection summary JSON."""
    summary_path = PROC_DIR / "drift_summary.json"
    if not summary_path.exists():
        return {"status": "not_run", "results": {}, "any_drift": False}
    with open(summary_path) as f:
        summary = json.load(f)
    any_drift = any(v.get("drift_detected", False) for v in summary.values())
    return {"status": "drift_detected" if any_drift else "no_drift",
            "any_drift": any_drift, "results": summary}


@router.get("/drift/report/{cls_name}")
async def get_drift_report_html(cls_name: str):
    """Returns Evidently HTML drift report for a class."""
    report_path = PROC_DIR / f"drift_report_{cls_name}.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report not found for {cls_name}")
    with open(report_path) as f:
        content = f.read()
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=content)


# ── 2. Model Comparison ───────────────────────────────────────
@router.get("/mlflow/runs")
async def get_mlflow_runs():
    """Returns all MLflow runs for model comparison."""
    try:
        import mlflow
        from backend.app.core.config import settings
        mlflow.set_tracking_uri("http://host.docker.internal:5005")
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        all_runs = []
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.val_f1 DESC"],
                max_results=20,
            )
            for run in runs:
                all_runs.append({
                    "run_id"      : run.info.run_id[:8],
                    "full_run_id" : run.info.run_id,
                    "status"      : run.info.status,
                    "start_time"  : datetime.fromtimestamp(run.info.start_time/1000).strftime("%Y-%m-%d %H:%M") if run.info.start_time else None,
                    "params"      : {
                        "batch_size"    : run.data.params.get("batch_size", "—"),
                        "epochs_phase1" : run.data.params.get("epochs_phase1", "—"),
                        "epochs_phase2" : run.data.params.get("epochs_phase2", "—"),
                        "lr_phase1"     : run.data.params.get("lr_phase1", "—"),
                        "model"         : run.data.params.get("model", "EfficientNetB0"),
                        "git_commit"    : run.data.params.get("git_commit", "—"),
                    },
                    "metrics"     : {
                        "val_f1"    : round(run.data.metrics.get("val_f1",    0), 4),
                        "val_acc"   : round(run.data.metrics.get("val_acc",   0), 4),
                        "test_f1"   : round(run.data.metrics.get("test_f1",   0), 4),
                        "test_acc"  : round(run.data.metrics.get("test_acc",  0), 4),
                        "test_loss" : round(run.data.metrics.get("test_loss", 0), 4),
                        "macro_auc" : round(run.data.metrics.get("macro_auc", 0), 4),
                    },
                    "experiment"  : exp.name,
                })
        return {"runs": all_runs, "total": len(all_runs)}
    except Exception as e:
        logger.error(f"MLflow runs fetch failed: {e}")
        return {"runs": [], "total": 0, "error": str(e)}


# ── 3. PDF Clinical Report ────────────────────────────────────
@router.post("/report/pdf")
async def generate_pdf_report(data: dict):
    """
    Generates a PDF clinical report for a scan result.
    Input: prediction result dict with patient details.
    Returns: PDF file as streaming response.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

        buffer = BytesIO()
        doc    = SimpleDocTemplate(buffer, pagesize=A4,
                    topMargin=2*cm, bottomMargin=2*cm,
                    leftMargin=2*cm, rightMargin=2*cm)

        styles  = getSampleStyleSheet()
        BLUE    = colors.HexColor("#0C2340")
        LBLUE   = colors.HexColor("#378ADD")
        RED     = colors.HexColor("#E24B4A")
        GREEN   = colors.HexColor("#639922")
        AMBER   = colors.HexColor("#BA7517")
        LGRAY   = colors.HexColor("#F0F4F8")

        title_style = ParagraphStyle("title", parent=styles["Title"],
            textColor=BLUE, fontSize=22, spaceAfter=4, fontName="Helvetica-Bold")
        sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
            textColor=LBLUE, fontSize=12, spaceAfter=2)
        label_style = ParagraphStyle("label", parent=styles["Normal"],
            textColor=colors.HexColor("#5A7A94"), fontSize=9,
            fontName="Helvetica", spaceAfter=2)
        body_style  = ParagraphStyle("body", parent=styles["Normal"],
            fontSize=10, leading=14, alignment=TA_JUSTIFY)
        disc_style  = ParagraphStyle("disc", parent=styles["Normal"],
            textColor=colors.HexColor("#888888"), fontSize=8,
            fontName="Helvetica-Oblique", alignment=TA_CENTER)

        story = []

        # Header
        story.append(Paragraph("RadiologyAI", title_style))
        story.append(Paragraph("AI-Assisted Chest X-Ray Clinical Report", sub_style))
        story.append(HRFlowable(width="100%", thickness=2, color=LBLUE))
        story.append(Spacer(1, 0.3*cm))

        # Patient info table
        pred_class = data.get("predicted_class", "Unknown")
        confidence = data.get("confidence", 0)
        risk       = data.get("risk_level", "Unknown")
        patient_id = data.get("patient_id", "—")
        age        = data.get("age", "—")
        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        patient_data = [
            ["Patient ID", patient_id,   "Date/Time",   timestamp],
            ["Age",        str(age),      "Report ID",   f"RAI-{datetime.now().strftime('%Y%m%d%H%M%S')}"],
            ["View",       "PA (Posteroanterior)", "Model", "EfficientNetB0"],
        ]
        pt = Table(patient_data, colWidths=[3.5*cm, 6*cm, 3.5*cm, 6*cm])
        pt.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), LGRAY),
            ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",    (2,0), (2,-1), "Helvetica-Bold"),
            ("TEXTCOLOR",   (0,0), (0,-1), BLUE),
            ("TEXTCOLOR",   (2,0), (2,-1), BLUE),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.white),
            ("PADDING",     (0,0), (-1,-1), 6),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[LGRAY, colors.white]),
        ]))
        story.append(pt)
        story.append(Spacer(1, 0.4*cm))

        # Diagnosis section
        story.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("AI DIAGNOSIS", label_style))

        diag_color = RED if pred_class != "Normal" else GREEN
        diag_data  = [[
            Paragraph(f"<b><font color='#{('%02x%02x%02x' % (int(diag_color.red*255), int(diag_color.green*255), int(diag_color.blue*255))).upper()}'>{pred_class}</font></b>", ParagraphStyle("d", fontSize=20, fontName="Helvetica-Bold")),
            Paragraph(f"Confidence: <b>{confidence*100:.1f}%</b>", ParagraphStyle("c", fontSize=11)),
            Paragraph(f"Risk Level: <b>{risk}</b>", ParagraphStyle("r", fontSize=11, textColor=diag_color)),
        ]]
        dt = Table(diag_data, colWidths=[6*cm, 5.5*cm, 5.5*cm])
        dt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.white),
            ("GRID",       (0,0), (-1,-1), 0.5, LGRAY),
            ("PADDING",    (0,0), (-1,-1), 10),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(dt)
        story.append(Spacer(1, 0.3*cm))

        # Class probabilities
        story.append(Paragraph("CLASS PROBABILITIES", label_style))
        probs = data.get("all_probabilities", [])
        if probs:
            prob_data = [["Class", "Confidence", "Risk"]]
            for p in probs:
                risk_map = {"Normal": "Low", "Pneumonia": "High", "COVID19": "High"}
                prob_data.append([p["class_name"], f"{p['confidence']*100:.2f}%",
                                  risk_map.get(p["class_name"], "—")])
            prob_table = Table(prob_data, colWidths=[6*cm, 5.5*cm, 5.5*cm])
            prob_table.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0), BLUE),
                ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
                ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",      (0,0), (-1,-1), 10),
                ("GRID",          (0,0), (-1,-1), 0.5, LGRAY),
                ("PADDING",       (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LGRAY]),
            ]))
            story.append(prob_table)
        story.append(Spacer(1, 0.3*cm))

        # Grad-CAM image
        gradcam_b64 = data.get("gradcam_base64")
        if gradcam_b64:
            story.append(Paragraph("GRAD-CAM EXPLAINABILITY HEATMAP", label_style))
            story.append(Paragraph(
                "The heatmap highlights regions of the chest X-ray that most influenced the AI prediction. "
                "Red/orange areas indicate high model attention.", body_style))
            story.append(Spacer(1, 0.2*cm))
            try:
                img_data = base64.b64decode(gradcam_b64)
                img_buf  = BytesIO(img_data)
                rl_img   = RLImage(img_buf, width=10*cm, height=6*cm)
                story.append(rl_img)
            except Exception as e:
                logger.warning(f"Grad-CAM image failed: {e}")
        story.append(Spacer(1, 0.3*cm))

        # Inference details
        story.append(Paragraph("INFERENCE DETAILS", label_style))
        inf_data = [
            ["Model",            "EfficientNetB0 (PyTorch 2.3.0)"],
            ["Inference Time",   f"{data.get('inference_time_ms', 0):.1f} ms"],
            ["Image Size",       "224 × 224 px"],
            ["Framework",        "PyTorch + Grad-CAM"],
            ["Report Generated", timestamp],
        ]
        inf_table = Table(inf_data, colWidths=[5*cm, 12*cm])
        inf_table.setStyle(TableStyle([
            ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE",  (0,0), (-1,-1), 9),
            ("TEXTCOLOR", (0,0), (0,-1), BLUE),
            ("GRID",      (0,0), (-1,-1), 0.5, LGRAY),
            ("PADDING",   (0,0), (-1,-1), 6),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[LGRAY, colors.white]),
        ]))
        story.append(inf_table)
        story.append(Spacer(1, 0.5*cm))

        # Disclaimer
        story.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "DISCLAIMER: This report is generated by an AI-assisted decision support system and is intended "
            "for preliminary assessment only. It is NOT a substitute for professional radiological diagnosis. "
            "All findings must be reviewed and confirmed by a qualified radiologist before any clinical decision is made.",
            disc_style))

        doc.build(story)
        buffer.seek(0)

        filename = f"RadiologyAI_Report_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        return StreamingResponse(
            buffer,
            media_type    = "application/pdf",
            headers       = {"Content-Disposition": f"attachment; filename={filename}"}
        )

    except ImportError:
        raise HTTPException(status_code=500, detail="reportlab not installed. Run: pip install reportlab")
    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
