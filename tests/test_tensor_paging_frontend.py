"""Optional browser checks: install playwright and its Chromium to run these."""

import os

import pytest
from symbolica import E
from symbolica.community.tensor import TensorExpression

pw = pytest.importorskip("playwright.sync_api")


def test_widget_replaces_pages_rejects_stale_replies_and_disposes():
    viewer = TensorExpression(E("+".join(f"x^{i}" for i in range(201)))).paged(
        page_size=25
    )
    import _spenso_paging

    with pw.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(
                **(
                    {"executable_path": os.environ["CHROMIUM_EXECUTABLE"]}
                    if "CHROMIUM_EXECUTABLE" in os.environ
                    else {}
                )
            )
        except pw.Error as error:
            pytest.skip(f"Chromium unavailable: {error}")
        page = browser.new_page()
        page.set_content('<div id="viewer"></div>')
        page.evaluate(
            """async ({esm, state}) => {
            const module = await import(URL.createObjectURL(new Blob([esm], {type:'text/javascript'})));
            const callbacks = new Set();
            window.messages = [];
            window.setPage = next => {state=next; callbacks.forEach(f=>f());};
            window.callbackCount = () => callbacks.size;
            window.dispose = module.default.render({el:document.querySelector('#viewer'), model:{
                get:()=>state, send:m=>messages.push(m),
                on:(_,f)=>callbacks.add(f), off:(_,f)=>callbacks.delete(f)
            }});
        }""",
            {"esm": _spenso_paging.WIDGET_ESM, "state": viewer._state()},
        )
        # Static exports cannot accidentally enable live navigation.
        assert page.get_by_role("button", name="Next", exact=True).is_disabled()
        view = page.evaluate("messages[0].view")
        page.evaluate("s=>setPage(s)", dict(viewer._state(), connected=view))
        next_button = page.get_by_role("button", name="Next", exact=True)
        for i in range(4):
            next_button.click()
            assert next_button.is_disabled()
            request = page.evaluate("messages.at(-1).request")
            before = page.locator("math *").count()
            # A reply to an older request must leave the pending page untouched.
            page.evaluate("s=>setPage(s)", dict(viewer._state(), request=f"{view}:0"))
            assert next_button.is_disabled()
            assert page.locator("math *").count() == before
            state = viewer._action({"action": "next"})
            page.evaluate("s=>setPage(s)", dict(state, request=request))
            assert page.locator("math").count() == 1
            assert page.locator("math *").count() < 1000
            assert len(viewer._cache) <= 3
        assert (
            page.locator(".math").evaluate("e=>getComputedStyle(e).maxHeight") == "none"
        )
        assert (
            page.locator("[data-spenso-math]").evaluate(
                "e=>getComputedStyle(e).overflowX"
            )
            == "visible"
        )
        assert page.get_by_label("Horizontal scroll").is_checked()
        assert "Omitted portions" not in page.locator(".status").inner_text()
        page.get_by_label("Horizontal scroll").uncheck()
        assert not page.locator(".math").evaluate(
            "e=>e.classList.contains('horizontal')"
        )
        page.get_by_label("Horizontal scroll").check()
        assert page.locator(".math").evaluate("e=>e.classList.contains('horizontal')")
        page.evaluate("dispose()")
        assert page.evaluate("callbackCount()") == 0
        assert page.evaluate("messages.at(-1).action") == "dispose"
        assert page.locator("math").count() == 0
        browser.close()
