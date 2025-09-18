# HFSS 자동화 스크립트 (IronPython 2.7 호환)
# -*- coding: utf-8 -*-

import ScriptEnv
import json
import os
import csv

# HFSS 초기화
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()

# 경로 설정
jobs_file = os.path.join("..", "data", "jobs_to_run.json")
results_dir = os.path.join("..", "data", "simulation_results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ---- CSV → Mask 로딩 ----
def load_mask_from_csv(path):
    mask = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            mask.append([int(x) for x in row])
    return mask

# ---- Job 실행 ----
def run_one_job(job):
    job_id = job["job_id"]
    csv_path = job["mask_csv"]

    # 프로젝트/디자인 초기화
    oProject = oDesktop.NewProject()
    oDesign = oProject.InsertDesign("HFSS", "Design{}".format(job_id), "DrivenModal", "")
    oEditor = oDesign.SetActiveEditor("3D Modeler")

    # 마스크 불러오기
    mask = load_mask_from_csv(csv_path)
    nrow, ncol = len(mask), len(mask[0])

    # IDC finger 생성
    unit = "um"
    pitch = 2  # 픽셀 크기 (2um × 2um)
    for r in range(nrow):
        for c in range(ncol):
            if mask[r][c] > 0:
                rect_name = "pix_r{}_c{}".format(r, c)
                oEditor.CreateRectangle(
                    [
                        "NAME:RectangleParameters",
                        "IsCovered:=", True,
                        "XStart:=", str(c * pitch) + unit,
                        "YStart:=", str(r * pitch) + unit,
                        "ZStart:=", "0" + unit,
                        "Width:=", str(pitch) + unit,
                        "Height:=", str(pitch) + unit,
                        "WhichAxis:=", "Z",
                    ],
                    [
                        "NAME:Attributes",
                        "Name:=", rect_name,
                        "Color:=", "(132 132 193)",
                        "Transparency:=", 0,
                    ]
                )

    # 포트 추가 (왼쪽/오른쪽)
    oEditor.CreateRectangle(
        [
            "NAME:RectangleParameters",
            "IsCovered:=", True,
            "XStart:=", "0um",
            "YStart:=", "0um",
            "ZStart:=", "0um",
            "Width:=", "1um",
            "Height:=", str(nrow * pitch) + unit,
            "WhichAxis:=", "Z",
        ],
        ["NAME:Attributes", "Name:=", "LumpedPort_L"]
    )
    oEditor.CreateRectangle(
        [
            "NAME:RectangleParameters",
            "IsCovered:=", True,
            "XStart:=", str(ncol * pitch) + unit,
            "YStart:=", "0um",
            "ZStart:=", "0um",
            "Width:=", "1um",
            "Height:=", str(nrow * pitch) + unit,
            "WhichAxis:=", "Z",
        ],
        ["NAME:Attributes", "Name:=", "LumpedPort_R"]
    )

    # 포트 정의
    oModule = oDesign.GetModule("BoundarySetup")
    oModule.AssignLumpedPort(
        [
            "NAME:PortL",
            "Objects:=", ["LumpedPort_L"],
            "DoDeembed:=", False,
            "RenormalizeAllTerminals:=", True,
            "Impedance:=", "50ohm"
        ]
    )
    oModule.AssignLumpedPort(
        [
            "NAME:PortR",
            "Objects:=", ["LumpedPort_R"],
            "DoDeembed:=", False,
            "RenormalizeAllTerminals:=", True,
            "Impedance:=", "50ohm"
        ]
    )

    # Radiation 경계
    oModule.AssignRadiation(
        [
            "NAME:Rad1",
            "Objects:=", ["Region"],
        ]
    )

    # 해석 Setup
    oAnalysis = oDesign.GetModule("AnalysisSetup")
    oAnalysis.InsertSetup("HfssDriven",
        [
            "NAME:Setup1",
            "Frequency:=", "10GHz",
            "PortsOnly:=", False,
            "MaxDeltaS:=", 0.02,
            "UseMatrixConv:=", True,
        ]
    )
    oAnalysis.InsertFrequencySweep("Setup1",
        [
            "NAME:Sweep1",
            "IsEnabled:=", True,
            "RangeType:=", "LinearStep",
            "RangeStart:=", "1GHz",
            "RangeEnd:=", "20GHz",
            "RangeStep:=", "0.05GHz"
        ]
    )

    # 결과 내보내기
    s2p_path = os.path.join(results_dir, "result_{0:04d}.s2p".format(job_id))
    oDesign.ExportNetworkData("Setup1", "Sweep1", 3, s2p_path, ["All"], False, 50)

# ---- 메인 ----
def main():
    with open(jobs_file, "r") as f:
        jobs = json.load(f)["jobs"]

    for job in jobs:
        run_one_job(job)

if __name__ == "__main__":
    main()
