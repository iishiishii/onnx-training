import { useContext, useEffect, useRef } from "react";
import { Box } from "@mui/material";

export function NiivuePanel({ nv, panelWidth }: any) {
  const canvas = useRef(null);

  useEffect(() => {
    async function fetchData() {
      const niivue = nv;
      niivue.attachToCanvas(canvas.current);
      // console.log("nv volumes ", volumes)
      // await niivue.loadVolumes(volumes);
      niivue.setSliceType(nv.sliceTypeAxial);
      // console.log("nv volumes ", niivue.onImageLoaded)
    }
    fetchData();
  }, []);

  return (
    <div className="niivue" style={{ width: panelWidth }}>
      <canvas
        ref={canvas}
      />
      <div
        id="intensity"
        style={{
          height: "30px",
          width: "200px",
          position: "fixed",
          bottom: "0",
          left: "0",
          color: "white",
          paddingTop: "6px",
          paddingLeft: "5px",
        }}
      >
        &nbsp;
      </div>
    </div>
  );
}
